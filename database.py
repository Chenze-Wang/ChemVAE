import sqlite3

import os
from datetime import datetime
import pandas as pd
import selfies as sf
from group_selfies import GroupGrammar
import pickle
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import tensor, Tensor
from torch.utils.data import Dataset, DataLoader
import warnings

path_to_dataset = '/data/chenze/ChemVAE/chemdb' # default location to look for the databases.
# path_to_dataset = os.path.join(path_to_dataset, 'chemdb')
path_to_dataset = os.path.expanduser(path_to_dataset)
# can be specified during initialization of ChemDB instance

# ChemDB is stored in a folder named after the database
# main_df is a pandas dataframe, containing every column. 
# column is pandas Series and stored as: [column_name].dtype.csv
import enum
from typing import Iterable, Callable
import atexit

class SQL_DType(enum.Enum):
    TEXT = enum.auto()
    REAL = enum.auto()
    BLOB = enum.auto()
    NULL = enum.auto()
    INTEGER = enum.auto()

    # def to_TEXT(obj) -> str:
    #     return str(obj)
    
    # def to_REAL(obj) -> float:
    #     return float(obj)
    
    # def to_INTEGER(obj) -> int:
    #     return int(obj)

    # def to_BLOB(obj) -> bytes:
    #     '''
    #     Only python built-in picklable objects are recommended
    #     '''
    #     return pickle.dumps(obj)

class ChemDB():
    def __init__(self, path_to_db: str = path_to_dataset):
        self.path_to_db = path_to_db
        self.db_file = os.path.join(path_to_db, 'main.db')
        self.con = sqlite3.connect(self.db_file)
        atexit.register(self.con.close)
        self.cur = self.con.cursor()
        self.__lock = True
    
    # this class maybe used in multiprocessing, add a lock to prevent accidental overwritting
    def check_safe_to_write(self):
        if self.__lock:
            raise Exception(f'Database {self.db_file} is locked, does not support overwritting data!')
    
    def lock(self):
        self.__lock = True

    def unlock(self):
        '''
        Run with care.
        '''
        self.__lock = False

    def load_column(self, 
                    table:str, /, 
                    *columns, 
                    order_by:str='id', 
                    ascending=True,
                    number:int=-1) -> pd.DataFrame:
        """Load a column of data from table

        Args:
            table (str): name of table in the database
            order_by (str, optional): key to determine the order of the samples, ascending. Defaults to 'id'.
            ascending: output in ascending order or otherwise
            number: if set to -1, select all entries. If 5, select the first 5 entries.

        Raises:
            ValueError: when designated column is not present.

        Returns:
            pd.DataFrame: DataFrame containing results.
        """        
        columns: tuple[str]
        print('loading columns: '+' '.join(columns))
        available = self.available_columns(table=table)
        for col in columns:
            col: str
            if col not in available:
                raise ValueError(f'Column {col} not available')
        col_line = ','.join(columns)
        order = 'ASC' if ascending else 'DESC'
        sql_cmd = f'SELECT id, {col_line} FROM {table} ORDER BY {order_by} {order} LIMIT {number}'
        df = pd.read_sql_query(sql=sql_cmd, con=self.con, index_col='id')
        return df

    def save_column(self, 
                    column_name:str, 
                    table:str, 
                    values: Iterable, 
                    dtype: SQL_DType|str):
        # make sure no accidental overwritting
        self.check_safe_to_write()
        if isinstance(dtype, str):
            assert SQL_DType[dtype]
        if isinstance(dtype, SQL_DType):
            dtype = str(dtype).strip('SQL_DType.')


        if column_name in self.available_columns(table=table):
            raise ValueError(f'column {column_name} already exists, cannot save')
        
        if dtype == 'BLOB':
            values = [pickle.dumps(v) for v in values]
        try:
            sql_create = f'ALTER TABLE {table} ADD COLUMN {column_name} {dtype}'
            self.cur.execute(sql_create)
            values = [(v, i) for i, v in enumerate(values, start=1)]
            sql_update = f'UPDATE {table} SET {column_name} = ? WHERE id = ?'
            self.cur.executemany(sql_update, values)
            self.con.commit()
        except sqlite3.Error as e:
            self.con.rollback()
            raise e
        
    def rename_column(self, table:str, column:str, new_name:str):
        self.check_safe_to_write()
        sql_rename = f'ALTER TABLE {table} RENAME COLUMN {column} TO {new_name}'
        try:
            self.cur.execute(sql_rename)
            self.con.commit()
        except sqlite3.Error as e:
            self.con.rollback()
            raise e
        
    def drop_column(self, table: str, column_name:str):
        self.check_safe_to_write()
        if column_name == 'id':
            raise ValueError('Cannot drop primary key \'id\'')
        columns = self.available_columns(table=table)
        if column_name not in columns:
            raise ValueError(f'Column {column_name} does not exist')
        columns.pop(column_name)

        if len(columns) == 0:
            raise Exception('Cannot drop the last column of a table, use drop table instead')
        columns_to_keep = ', '.join(columns.keys())
        # Generate the SQL commands
        create_new_table_sql = f"""
        CREATE TABLE {table}_new AS 
        SELECT {columns_to_keep}
        FROM {table};"""
        drop_old_table_sql = f"DROP TABLE {table};"
        rename_table_sql = f"ALTER TABLE {table}_new RENAME TO {table};"
        try:
            self.cur.execute(create_new_table_sql)
            self.cur.execute(drop_old_table_sql)
            self.cur.execute(rename_table_sql)
            self.con.commit()
        except Exception as e:
            self.con.rollback()
            raise e

    def available_columns(self, table: str) -> dict[str:str]:
        '''
        Return a dict of {column_name: dtype}
        '''
        res = self.cur.execute(f'''PRAGMA table_info({table})''').fetchall()
        res: list[tuple] # [(col_id, name, dtype, ...)]
        res:dict[str:str] = {t[1]:t[2] for t in res}
        # print(res)
        res.pop('id')
        return res
    
    def build_alphabet_atoi(self,
                      table: str,
                      column:str,
                      padding_token:str='[nop]',
                      tokenizer=list) -> tuple[list, dict]:
        '''
        Return and store an alphabet(convert index to symbol) and atoi(convert symbol to index)
        '''
        self.check_safe_to_write()
        columns_dtype = self.available_columns(table=table)
        if columns_dtype[column] != 'TEXT':
            raise ValueError('Cannot tokenizer data other than TEXT type')
        
        file_name = f'{table}.{column}.appendices.json'
        file_name = os.path.join(self.path_to_db, file_name)
        
        alphabet = set()
        l = self.load_column(table, column)[column].tolist()
        print(len(l))
        for s in l:
            alphabet = alphabet.union(set(tokenizer(s)))
        
        # make sure padding token is at the first place
        if padding_token in alphabet:
            alphabet.remove(padding_token)

        alphabet = list(alphabet)
        alphabet.insert(0, padding_token)
        # alphabet is a list, inverse it to form an atoi dict
        atoi = {ch: i for i, ch in enumerate(alphabet)}

        if os.path.isfile(file_name):
            # if appendices already exists, update it
            with open(file_name, 'r') as f:
                appendices:dict = json.load(f)
            appendices['alphabet'] = alphabet
            appendices['atoi'] = atoi
            appendices['alphabet_timestamp'] = datetime.now().isoformat()
        else:
            appendices = {'alphabet': alphabet, 'atoi': atoi, 'alphabet_timestamp': datetime.now().isoformat()}

        with open(file_name, 'w') as f:
            json.dump(appendices, f)
        
        return alphabet, atoi
    
    def load_appendices(self, table:str, column:str) -> dict:
        file_name = f'{table}.{column}.appendices.json'
        file_name = os.path.join(self.path_to_db, file_name)
        with open(file_name, 'r') as f:
            appendices = json.load(f)
        return appendices
        
    # when using torch.nn.Embedding, only indices of tokens should be fed in, 
    # so the real one-hot embeddings are not provided, only the sequence of indices
    def load_idxv(self, table:str, column:str, tokenizer=list, use_cache=True) -> list[Tensor]:
        '''
        the tokenizer is, by default, list (tokenizer char by char)

        NOTE: specifying use_cache=True will deprecate argument tokenizer

        cache will be a dict consisting of 'idxv': [tensor, tensor...], 'alphabet_timestamp': isoformat time string
        '''
        cache_file = os.path.join(self.path_to_db, f'{table}.{column}.idxv.pkl')

        app_file = f'{table}.{column}.appendices.json'
        app_file = os.path.join(self.path_to_db, app_file)
        
        if not os.path.isfile(app_file):
            raise Exception(f'Appendices file not found for {table}.{column}, unlock and call build_alphabet_atoi first.')
        
        with open(app_file, 'rb') as f:
            appendices = json.load(f)
        
        # only if use_cache and cache exists, use cache
        if use_cache and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f) # keys: idxv, alphabet_timestamp
            if cache['alphabet_timestamp'] != appendices['alphabet_timestamp']:
                raise Exception('idxv cache incongruent with current alphabet. Set use_cache=False and build idxv again')
            return cache['idxv']
        
        else:
            str_list = self.load_column(table, column)[column].tolist()
            if os.path.isfile(cache_file):
                print(f'rewriting idxv cache file for {table}.{column}')
            else:
                print(f'idxv cache for {table}.{column} not found, building')
            # build the one hot embeddings
            alphabet:list[str] = appendices['alphabet']
            atoi:dict[str:int] = appendices['atoi']
            
            # function to turn a string into index vector
            def str2idxv(s:str, tokenizer, atoi):
                return torch.tensor([atoi[a] for a in tokenizer(s)], dtype=torch.long)
            idxv = [str2idxv(s, tokenizer=tokenizer, atoi=atoi) for s in str_list]

            with open(cache_file, 'wb') as f:
                pickle.dump({'idxv':idxv, 'alphabet_timestamp':appendices['alphabet_timestamp']}, f)

            return idxv


class IDXVset(Dataset):
    def __init__(self, idxv: Tensor|list[Tensor]):
        '''
        Dataset class for padded idxv!!!
        '''
        super().__init__()
        if isinstance(idxv, list):
            idxv = pad_sequence(idxv, batch_first=True, padding_value=0)
        assert idxv.dtype is torch.long
        self.idxv = idxv
    
    def __getitem__(self, i):
        return self.idxv[i]
    
    def __len__(self):
        return self.idxv.shape[0]
    
    def shuffle(self, seed:int=1024):
        rng = np.random.RandomState(seed=seed)
        shuf = rng.permutation(self.idxv.shape[0])
        self.idxv = self.idxv[shuf]
    
    def slice(self, i: slice):
        '''
        Return a slice of itself.
        '''
        return IDXVset(self.idxv[i])

class IDXV_prop(Dataset):
    def __init__(self, 
                 idxv: Tensor|list[Tensor], 
                 properties: pd.DataFrame|np.ndarray|Tensor, 
                 prop_dtype=torch.float32,
                 standardize_props = True):
        '''
        Dataset class for padded idxv!!!
        '''
        super().__init__()
        if isinstance(idxv, list):
            idxv = pad_sequence(idxv, batch_first=True, padding_value=0)
        assert idxv.dtype is torch.long
        self.idxv = idxv

        if isinstance(properties, pd.DataFrame):
            self.prop_tensor: tensor = tensor(properties.values, dtype=prop_dtype)
        elif isinstance(properties, np.ndarray):
            self.prop_tensor = tensor(properties, dtype=prop_dtype)
        elif isinstance(properties, Tensor):
            self.prop_tensor = properties.to(dtype=prop_dtype)
        else:
            raise Exception(f'type {type(properties)} for argument properties not supported.')

        self.has_standardized = False
        if standardize_props:
            self.prop_mean = torch.mean(self.prop_tensor, dim=0, keepdim=True)
            self.prop_std = torch.std(self.prop_tensor, dim=0, keepdim=True)
            self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std
            self.has_standardized = True

    def __getitem__(self, i: int):
        return self.idxv[i], self.prop_tensor[i]
    
    def standardize_properties(self, 
                               mean:Tensor|None=None, 
                               std:Tensor|None=None):
        """Standardize the properties. 
        If arguments are provided, standadize the properties with specified mean and std.

        Args:
            mean (Tensor, optional): Must 1-D or shape of (1, data_size) Defaults to None.
            std (Tensor, optional): Must 1-D or shape of (1, data_size) Defaults to None.
        """        
        if mean is None:
            assert std is None
            self.prop_mean = torch.mean(self.prop_tensor, dim=0, keepdim=True)
            self.prop_std = torch.std(self.prop_tensor, dim=0, keepdim=True)
            self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std
            self.has_standardized = True
            return
        
        assert std is not None
        if len(mean.shape) == 1:
            assert len(std.shape) == 1
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        else:
            assert (len(mean.shape) == 2) and (len(std.shape) == 2)
            assert (mean.shape[0] == 1) and (std.shape[0] == 1)
        
        assert torch.all(std>0.0)
        
        self.prop_mean = mean
        self.prop_std = std
        self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std
                
    def __len__(self):
        return self.idxv.shape[0]


class IDXVv_prop(Dataset):
    def __init__(self,
                idxv_list: list[Tensor], 
                properties: pd.DataFrame|np.ndarray|Tensor, 
                prop_dtype=torch.float32,
                standardize_props = True):
        """Dataset class for unpadded sequences
        'IDXVv', v stands for variable length

        """        
        super().__init__()
        
        if isinstance(idxv_list, list) and isinstance(idxv_list[0], Tensor):
            pass
        else:
            raise ValueError(f'input type of idxv is not supported. check the entries in it')
        self.idxv_list = idxv_list
        if isinstance(properties, pd.DataFrame):
            self.prop_tensor: tensor = tensor(properties.values, dtype=prop_dtype)
        elif isinstance(properties, np.ndarray):
            self.prop_tensor = tensor(properties, dtype=prop_dtype)
        elif isinstance(properties, Tensor):
            self.prop_tensor = properties.to(dtype=prop_dtype)
        else:
            raise Exception(f'type {type(properties)} for argument properties not supported.')
        
        self.has_standardized = False
        if standardize_props:
            self.prop_mean = torch.mean(self.prop_tensor, dim=0, keepdim=True)
            self.prop_std = torch.std(self.prop_tensor, dim=0, keepdim=True)
            self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std
            self.has_standardized = True

    def __getitem__(self, i: int):
        return (self.idxv_list[i], self.prop_tensor[i])
        
    def __len__(self):
        return len(self.idxv_list)
    
    def standardize_properties(self, 
                               mean:Tensor|None=None, 
                               std:Tensor|None=None):
        """Standardize the properties. 
        If arguments are provided, standadize the properties with specified mean and std.

        Args:
            mean (Tensor, optional): Must 1-D or shape of (1, data_size) Defaults to None.
            std (Tensor, optional): Must 1-D or shape of (1, data_size) Defaults to None.
        """        
        if mean is None:
            assert std is None
            self.prop_mean = torch.mean(self.prop_tensor, dim=0, keepdim=True)
            self.prop_std = torch.std(self.prop_tensor, dim=0, keepdim=True)
            self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std
            self.has_standardized = True
            return
        
        assert std is not None
        if len(mean.shape) == 1:
            assert len(std.shape) == 1
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        else:
            assert (len(mean.shape) == 2) and (len(std.shape) == 2)
            assert (mean.shape[0] == 1) and (std.shape[0] == 1)
        
        assert torch.all(std>0.0)
        
        self.prop_mean = mean
        self.prop_std = std
        self.prop_tensor = (self.prop_tensor - self.prop_mean) / self.prop_std

    @staticmethod
    def _collate_fn(batch: tuple):
        # receives what __getitem__ returns
        # batch is a list of tuples (idxv:tensor, prop: tensor)
        seqs = []
        props = []
        for x, y in batch:
            seqs.append(x)
            props.append(y)

        # Pad sequences
        padded_sequences = pad_sequence(seqs, batch_first=True, padding_value=0)
        props = torch.vstack(props)
        return padded_sequences, props

# generate pytorch dataloaders from idxvs and properties
def get_loaders(idxv:Tensor|list[Tensor], 
                properties:Tensor|np.ndarray|pd.DataFrame, 
                batch_size: int, 
                split:float = 0.8,
                seed:int=1024,
                prop_dtype:torch.dtype=torch.float,
                mode:str='equal_length') -> tuple[list[DataLoader], Tensor, Tensor]:
    """Constructs train and eval DataLoaders from raw data.

    Args:
        idxv (Tensor | list[Tensor]): The index vector tensor, dtype is torch.long
            Or a list of idxv Tensors
            
        properties (Tensor | np.ndarray): the property values, corresponding to each idxv. 
            *Note: no need to standardize ahead, this function handles the it.
            
        batch_size (int): batch size for both the train loader, and the eval loader
        
        split float: portion of the training set. Must be strictly smaller than 1.0
            
        seed: the seed to generate random partition of data
        
        prop_dtype: the data type to use on the properties
        
        mode: choose between "equal_length" and "variable_length". 
            "equal_length" mode: we pad all the idxv so that they all have same length as the longest sample.
            "variable_length" mode: we keep samples of variable length in a list, 
                only pad they to equal length in collate_fn.

    Raises:
        ValueError: If the split do not sum up to 1.
        Exception: If the portion in split is too small that the size of its data set becomes 0.

    Returns:
        list[DataLoader]: loader_train and loader_val
        Tensor, Tensor: mean and std of properties in the training set
    """
    assert len(idxv) == len(properties)
    assert idxv[0].dtype is torch.long
    
    total_num = len(idxv)
    
    if not (0.0<split<1.0):
        raise ValueError('split should be a fraction between 0.0 and 1.0')
    
    if isinstance(properties, pd.DataFrame):
        prop_tensor = tensor(properties.values, dtype=prop_dtype)
    elif isinstance(properties, np.ndarray):
        prop_tensor = tensor(properties, dtype=prop_dtype)
    elif isinstance(properties, Tensor):
        prop_tensor = properties.to(dtype=prop_dtype)
    else:
        raise ValueError(f'type {type(properties)} for argument [properties] not supported.')
    
    '''
    def shuffle(self, seed:int=1024):
        rng = np.random.RandomState(seed=seed)
        shuf = rng.permutation(self.idxv.shape[0])
        self.idxv = self.idxv[shuf]
        self.prop_tensor = self.prop_tensor[shuf]
    '''
    # shuffle the raw samples
    rng = np.random.RandomState(seed=seed)
    shuffle = rng.permutation(total_num)
    if isinstance(idxv, Tensor):
        shuffled_idxv = idxv[shuffle]
    elif isinstance(idxv, list):
        shuffled_idxv = [idxv[i] for i in shuffle]
    else:
        raise ValueError((f'type {type(idxv)} for argument [idxv] not supported.'))
        
    shuffled_prop_tensor = prop_tensor[shuffle]
    
    num_train = int(total_num*split)
    if (num_train==0) or (num_train==total_num):
        raise Exception(f'split {split} is too close to 0 or 1. Less than one sample in train/val set')

    idxv_train = shuffled_idxv[:num_train]
    idxv_val = shuffled_idxv[num_train:]
    prop_train = shuffled_prop_tensor[:num_train]
    prop_val = shuffled_prop_tensor[num_train:]
    loaders = []
    start = 0
    if mode == 'variable_length':
        train_set = IDXVv_prop(idxv_list=idxv_train, 
                               properties=prop_train, 
                               prop_dtype=prop_dtype, 
                               standardize_props=True)
        
        loaders.append(DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=IDXVv_prop._collate_fn,
                                  num_workers=16))
        
        val_set = IDXVv_prop(idxv_list=idxv_val, 
                             properties=prop_val, 
                             prop_dtype=prop_dtype, 
                             standardize_props=False)
        
        # standardize the val set with training mean and std.
        val_set.standardize_properties(train_set.prop_mean, train_set.prop_std)
        loaders.append(DataLoader(val_set, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=IDXVv_prop._collate_fn,
                                  num_workers=16))
    elif mode=='equal_length':
        train_set = IDXV_prop(idxv=idxv_train, 
                               properties=prop_train, 
                               prop_dtype=prop_dtype, 
                               standardize_props=True)
        
        loaders.append(DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=16))
        
        val_set = IDXV_prop(idxv=idxv_val, 
                             properties=prop_val, 
                             prop_dtype=prop_dtype, 
                             standardize_props=False)
        
        # standardize the val set with training mean and std.
        val_set.standardize_properties(train_set.prop_mean, train_set.prop_std)
        loaders.append(DataLoader(val_set, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=16))
    else:
        raise Exception(f'mode {mode} is not supported')
        
    
    return loaders, train_set.prop_mean, train_set.prop_std


if __name__=='__main__':
    cdb = ChemDB()
    table = 'qm9'
    # cdb.unlock()
    print(cdb.available_columns(table=table))
    # alp, atoi = cdb.build_alphabet_atoi(table=table, column='SELFIES', tokenizer=sf.split_selfies)
    appendices = cdb.load_appendices(table=table, column='SELFIES')
    atoi = appendices['atoi']
    idxv = cdb.load_idxv(table=table, column='SELFIES', tokenizer=sf.split_selfies)[:10]
    df = cdb.load_column(table, 'SELFIES', number=10)

    prop = cdb.load_column(table, 'logP', 'QED', number=10)
    
    print(df.tail())
    print(prop.tail())
    
    print(idxv[-5:])
    print(atoi)
    
    (loader_train, loader_val), mean, std = get_loaders(idxv, prop, 3, 0.6, mode='variable_length')
    for x, y in loader_train:
        print(x)
        print(y*std + mean)
        break
    
# cur.execute('''CREATE TABLE qm9(
#     id INTEGER PRIMARY KEY,
#     SMILES TEXT NOT NULL UNIQUE,
#     SELFIES TEXT,
#     Weight REAL, 
#     logP REAL,
#     QED REAL, 
#     SAS REAL, 
#     TPSA REAL, 
#     NumHAcceptors REAL, 
#     NumHDonors REAL, 
#     NumRotatableBonds REAL, 
#     NumAromaticRings REAL, 
#     FractionCSP3 REAL, 
#     HeavyAtomCount REAL, 
#     MolMR REAL)''')

# with open('/data/chenze/ChemVAE/datasets/0SelectedSMILES_QM9.txt', 'r') as f:
#     n = 0
#     for line in f:
#         # if n > 1000:
#         #     break
#         if line.startswith('idx'):
#             continue
#         smiles = line.split(',')[1]
#         smiles = smiles.strip()
#         print(smiles)
#         selfies = sf.encoder(smiles)
#         mol = Chem.MolFromSmiles(smiles)
#         props = [func(mol) for func in funcs.values()]
#         insert_cmd = \
#         '''
#         INSERT INTO qm9(SMILES, SELFIES, Weight, logP, QED, SAS, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAromaticRings, FractionCSP3, HeavyAtomCount, MolMR)
#         VALUES({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'''
#         # print(insert_cmd.format(smiles, selfies, *props))
#         cur.execute(insert_cmd.format('\''+smiles+'\'', '\''+selfies+'\'', *props))

#         n += 1
# con.commit()