import os
pwd = os.path.dirname(__file__)
import time
from datetime import datetime
import logging
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import chem_utils as cu
import TransformerVAE as TVAE

from group_selfies import GroupGrammar

from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Function {func.__name__} took {run_time:.4f} seconds to execute")
        return result
    return wrapper
    
SAVE_CHECKPOINT_EVERY = 10

class ChemVAETrainer:
    def __init__(self,
                 vae:TVAE.TransformerVAE,
                 loaders: dict[str: DataLoader],
                 hyper_params: dict[str: int|dict],
                 sup_params: dict,
                 logger:Logger,
                 device: torch.device):
        assert 'alphabet' in sup_params.keys()
        assert 'type_of_encoding' in sup_params.keys()
        assert 'max_seq_len' in sup_params.keys()
        
        if 'group_grammar' in sup_params.keys():
            gg = GroupGrammar.from_file(self.sup_params['group_grammar']) | GroupGrammar.essential_set()
            self.to_smiles = lambda grp_SLFS: cu.group_selfies2smiles(grp_SLFS, gg)
        else:
            self.to_smiles = cu.selfies2smiles

        if 'l1_reg_lambda' in hyper_params.keys():
            self.l1_reg_lambda = hyper_params['l1_reg_lambda']
            
            if 'group_dim' in hyper_params:
                assert vae.latent_dim%hyper_params['group_dim']==0
                self.group_dim = hyper_params['group_dim']
            else:
                self.group_dim = 1
        else:
            self.l1_reg_lambda = None
            self.group_dim = None

        self.hyper_params = hyper_params
        self.sup_params = sup_params
        self.vae = vae
        self.loaders = loaders
        self.device = device
        self.logger = logger
        self.current_epoch = 0

    def _compute_recon_quality(self, 
                               x_hat: torch.Tensor, 
                               x: torch.Tensor) -> tuple[float, float]:
        '''
        compute both the recon. quality and mol. acc.
        Args
            x: a batch of index vectors (batch, seq) : ground truths
            
            x_hat: the predicted index vectors from decoder (batch, seq)
        
        Return
            (float, float)
        '''
        is_right = (x==x_hat)
        recon_quality = torch.mean(is_right.to(dtype=torch.float32)).item()
        sample_acc = torch.mean(torch.all(is_right, dim=1).to(dtype=torch.float32)).item()
        return  recon_quality, sample_acc

    # return the quality rate and number of total correct molecules
    def _latent_space_quality(self, sample_num: int, max_len: int) -> tuple[float, float]:
        '''
        sample a batch of latent points

        Args
            sample_num: the number of samples

            max_len: maximal length of decoded sequence

        Return
            validity, uniqueness(diversity)
        
        '''
        sampled_latent_points = torch.randn(sample_num, 
                                            self.vae.encoder.get_latent_dim(), 
                                            device=self.device)
        
        decoded = self.vae.decode_autoregressive(sampled_latent_points, 
                                                 max_len=max_len, 
                                                 padding_idx=0,
                                                 stop_on_pad=True)
        strings = cu.idxv2str(decoded,
                              alphabet=self.sup_params['alphabet'])
        strings = [self.to_smiles(s) for s in strings]

        return cu.validity_diversity(strings)
        

    def _quality_in_valid_set(self) -> tuple[float, float, float]:
        '''
        compute the recon. quality, mol. acc., prop_mse on the validation set
        '''
        recon_quality_list = []
        mol_acc_list = []
        prop_mse_list = []
        
        for x, y in self.loaders['val']:
            x: torch.Tensor
            y: torch.Tensor

            x = x.to(self.device) # (batch, seq_len)
            y = y.to(self.device) # (batch, num_props)
            
            latent_points, _, _ = self.vae.encode(x) # latent_points (batch, latent_dim)
            prop_mse = nn.functional.mse_loss(self.vae.latent_regressor.forward(latent_points), y)
            prop_mse_list.append(prop_mse.item())
            
            x_hat = self.vae.decode_autoregressive(z=latent_points, 
                                                     max_len=self.sup_params['max_seq_len'],
                                                     stop_on_pad=False)
            recon, mol_acc = self._compute_recon_quality(x_hat, x)
            recon_quality_list.append(recon)
            mol_acc_list.append(mol_acc)

        return np.mean(recon_quality_list).item(), np.mean(mol_acc_list).item(), np.mean(prop_mse_list).item()
    
    def evaluate(self) -> dict:
        # 'latent_val', 'latent_div', 
        metrics = ['latent_val', 'latent_div', 'recon', 'mol_acc', 'prop_mse']
        lv, ld = self._latent_space_quality(1000, self.sup_params['max_seq_len'])
        rc, ma, pm = self._quality_in_valid_set()
        # lv, ld, 
        vals = [lv, ld, rc, ma, pm]
        return {metric: val for metric, val in zip(metrics, vals)}
    
    # dict is of {metric_name, metric_value}
    def generate_summary(self, stats: dict):
        return ' | '.join(f'{key}: {val:.4f}' for key, val in stats.items())
    
    def save_checkpoint(self, timestamp_tname:str, dir:str):
        '''
        save model state_dict to dir
        '''
        assert self.current_epoch > 0
        checkpoint = os.path.join(dir, f'{timestamp_tname}.{self.current_epoch}.checkpoint.pt')
        torch.save(self.vae.state_dict(), checkpoint)

    def load_checkpoint(self, timestamp_tname:str, epoch:int, dir:str):
        checkpoint = os.path.join(dir, f'{timestamp_tname}.{epoch}.checkpoint.pt')
        if not os.path.isfile(checkpoint):
            raise Exception(f'checkpoint {checkpoint} not found')
        self.vae.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.current_epoch = epoch

    def train(self, timestamp_tname:str, path_to_checkpoints:str):
        self.vae.to(device=self.device)
            
        # initialize optimizer
        self.optimizer = Adam(self.vae.parameters(), **self.hyper_params['optimizer_args'])

        if 'plateau_reduce_criteria' in self.sup_params.keys():
            prc = self.sup_params['plateau_reduce_criteria']
            self.plateau_reducer = ReduceLROnPlateau(self.optimizer, mode='max', patience=prc['patience'], factor=prc['factor'])

        start = time.time()
        loader_train: DataLoader = self.loaders['train']
        history_early_stop = []
        for e in range(self.hyper_params['max_epochs']):
            
            self.vae.train()

            for x, y in loader_train:
                x: torch.Tensor
                y: torch.Tensor

                x = x.to(self.device)
                y = y.to(self.device)
                loss: torch.Tensor = self.vae.training_loss(x, y=y, 
                                              kld_beta=self.hyper_params['kld_beta'],
                                              prop_loss_alpha=self.hyper_params['prop_loss_alpha'],
                                              l1_reg_lambda=self.l1_reg_lambda,
                                              group_dim=self.group_dim)
                self.optimizer.zero_grad()
                loss.backward()
                if 'grad_norm_clip' in self.sup_params:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 
                                                   self.sup_params['grad_norm_clip'])
                self.optimizer.step()

            self.current_epoch += 1

            if ((e+1)%self.sup_params['evaluate_every']==0) or \
                ((e+1)==self.hyper_params['max_epochs']):
                
                self.vae.eval()
                stats = self.evaluate()
                stats['epoch'] = e
                stats['time'] = time.time() - start
                start = time.time()
                self.logger.info(self.generate_summary({key: np.mean(val) for key, val in stats.items()}))
                
                if 'plateau_reduce_criteria' in self.sup_params.keys():
                    self.plateau_reducer.step(stats[self.sup_params['plateau_reduce_criteria']['metric']])
                    # lrs = [i['lr'] for i in optimizer.param_groups]
                    # self.logger.info(f'current lr: {lrs}')

                if 'early_stopping_criteria' in self.sup_params.keys():
                    # check is early stopping is needed
                    es_metric = self.sup_params['early_stopping_criteria']['metric']
                    patience = self.sup_params['early_stopping_criteria']['patience']
                    history_early_stop.append(stats[es_metric])
                    if (len(history_early_stop) - np.argmax(history_early_stop)) > patience:
                        self.logger.info('Early stopping criteria')
                        return
            
            if (self.current_epoch)%SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(timestamp_tname=timestamp_tname, dir=path_to_checkpoints)

from omegaconf import OmegaConf, DictConfig
import database
def run_task(timestamp:str, task_name: str, device: torch.device|str|int):

    paths:DictConfig = OmegaConf.load(os.path.join(pwd, 'path.yml'))

    task_name = task_name if task_name.endswith('.yml') else (task_name+'.yml')
    task_config:DictConfig = OmegaConf.load(os.path.join(paths['path_to_task_config'], task_name))

    # load data here
    cdb = database.ChemDB(paths['path_to_database'])
    dataset_name = task_config['dataset_name']
    type_of_encoding = task_config['sup_params']['type_of_encoding']

    idxv = cdb.load_idxv(dataset_name, column=type_of_encoding)
    appendices = cdb.load_appendices(dataset_name, type_of_encoding)
    alphabet = appendices['alphabet']
    properties = cdb.load_column(dataset_name, *task_config['properties'])
    
    dataset_whole = database.IDXV_prop(idxv=idxv, properties=properties, normalize_props=False)
    train_ratio, _ = task_config['train_val_split']
    dataset_whole.shuffle()
    train_set = dataset_whole.slice(slice(None, int(train_ratio*len(dataset_whole))))
    val_set = dataset_whole.slice(slice(int(train_ratio*len(dataset_whole)), None))
    
    loader_train = 
    loaders: dict[str: DataLoader] = {'train':loader_train, 'val': loader_val}

    # load the model configs
    model_config: str = task_config['model_config']
    model_config = model_config if model_config.endswith('.yml') else (model_config+'.yml')
    model_config = OmegaConf.load(os.path.join(paths['path_to_model_config'], model_config))
    
    # update some configs according to dataset
    OmegaConf.update(model_config, 'num_properties', len(task_config['properties']))
    OmegaConf.update(model_config, 'embedding_layer_args.num_embeddings', len(alphabet))
    OmegaConf.update(task_config, 'sup_params.alphabet', alphabet)
    OmegaConf.update(task_config, 'sup_params.max_seq_len', pt_dataset.idxv.shape[1])
    
    if model_config['encoder']['backbone'] == 'transformer_encoder_flatten':
        OmegaConf.update(model_config, 'encoder.model_args.seq_len', pt_dataset.idxv.shape[1])

    vae = TVAE.model_config_parser(model_config)

    task_name = task_name.strip('.yml')
    timestamp_tname = f'{timestamp}.{task_name}'
    logger = logging.getLogger(task_name)
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # Set up logging to your specific file
    log_file = os.path.join(paths['path_to_log'], f'{timestamp_tname}.log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(levelname)s.%(asctime)s: %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info(task_config['title'])
    logger.info(timestamp_tname)

    hyper_params:dict = task_config['hyper_params']
    sup_params:dict = task_config['sup_params']
    if not isinstance(device, torch.device):
        device = torch.device(device)
    trainer = ChemVAETrainer(vae=vae, 
                             loaders=loaders,
                             hyper_params=hyper_params,
                             sup_params=sup_params,
                             logger=logger,
                             device=device)

    task_config_yml = OmegaConf.to_yaml(task_config, resolve=True)
    model_config_yml = OmegaConf.to_yaml(model_config, resolve=True)
    
    # cache the model config, in case it changes or is lost
    with open(os.path.join(paths['path_to_results'], f'{timestamp_tname}.task_config.yml'), 'w') as f:
        f.write(task_config_yml)

    with open(os.path.join(paths['path_to_results'], f'{timestamp_tname}.model_config.yml'), 'w') as f:
        f.write(model_config_yml)

    logger.info('\n'+task_config_yml) # now in training log, the task config is shown instead of model config
    trainer.train(timestamp_tname=timestamp_tname, path_to_checkpoints=paths['path_to_checkpoints'])

    torch.save(vae.state_dict(), 
               os.path.join(paths['path_to_results'], f'{timestamp_tname}.vae.state_dict.pt'))
    
    logger.info('Finished...')


def get_pretrained_vae(timestamp:str, task_name: str|None=None) -> TVAE.TransformerVAE:
    if task_name is None:
        timestamp_tname = timestamp
    else:
        timestamp_tname = f'{timestamp}.{task_name}'

    paths = OmegaConf.load(os.path.join(pwd, 'path.yml'))
    path_to_results = paths['path_to_results']
    
    model_checkpoint = os.path.join(path_to_results, f'{timestamp_tname}.vae.state_dict.pt')
    model_config = os.path.join(path_to_results, f'{timestamp_tname}.model_config.yml')
    
    model_checkpoint:dict = torch.load(model_checkpoint, map_location='cpu', weights_only=True)
    model_config = OmegaConf.load(model_config)

    vae = TVAE.model_config_parser(model_config)
    vae.load_state_dict(model_checkpoint)
    return vae

def get_supplementaries(timestamp:str, task_name: str|None=None) -> dict:
    
    paths = OmegaConf.load(os.path.join(pwd, 'path.yml'))
    if task_name is None:
        timestamp_tname = timestamp
    else:
        timestamp_tname = f'{timestamp}.{task_name}'
        
    task_config = OmegaConf.load(os.path.join(paths['path_to_results'], f'{timestamp_tname}.task_config.yml'))
    type_of_encoding = task_config['sup_params']['type_of_encoding']
    dataset_name = task_config['dataset_name']
    alphabet = task_config['sup_params']['alphabet']
    max_seq_len = task_config['sup_params']['max_seq_len'] # longest sample in the data set
    
    model_config = OmegaConf.load(os.path.join(paths['path_to_results'], f'{timestamp_tname}.task_config.yml'))
    properties_used = 
    # get the mean and std of the properties
    
    return {'type_of_encoding': type_of_encoding,
            'alphabet': alphabet,
            'dataset_name': dataset_name,
            'max_seq_len': max_seq_len}


# unit testing: fitting QM9
if __name__ == '__main__':
    # start training with one line of code.
    run_task(timestamp=datetime.now().isoformat(), task_name='T0', device='cuda:0')