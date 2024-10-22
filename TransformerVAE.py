from abc import abstractmethod

import torch
from torch import nn
from torch.nn.functional import cross_entropy


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim))
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class AdditiveAttentionAggregator(nn.Module):
    def __init__(self, nhead:int, head_dim:int, mem_size_factor:int=4, ffn_hidden_size_factor:int=4):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.mem_size_factor = mem_size_factor
        self.ffn_hidden_size_factor = ffn_hidden_size_factor
        
        self.q_aggregator = nn.Parameter(torch.randn(1, 1, nhead, head_dim*mem_size_factor))
        
        size_w11 = head_dim*(mem_size_factor+1)
        size_w12 = size_w11*ffn_hidden_size_factor
        self.ffn_w1 = nn.Parameter(torch.randn(1, 1, nhead, size_w11, size_w12))
        self.ffn_b1 = nn.Parameter(torch.zeros(1, 1, nhead, size_w12))
        self.ffn_activation = nn.ReLU()
        self.ffn_w2 = nn.Parameter(torch.randn(1, 1, nhead, size_w12, 1))
        self.ffn_b2 = nn.Parameter(torch.zeros(1, 1, nhead, 1))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Aggregate an encoded sequence using additive attention

        Args:
            x (torch.Tensor): input sequence of shape (batch, seq, emb)

        Returns:
            torch.Tensor: attention weights to aggregate a sequence of embeddings
                shape (bs, seq, nhead, 1)
        """
        
        # x (batch, seq, emb), reshape into heads
        bs, seq_len, emb = x.shape
        x = x.reshape(bs, seq_len, self.nhead, self.head_dim) # (bs, seq, nhead, hdim)
        
        # self.q_aggregator (1, 1, nhead, head_dim*mem_size_factor)
        q_aggregator = self.q_aggregator.expand(bs, seq_len, -1, -1)
        concat = torch.cat([x, q_aggregator], dim=-1).unsqueeze(-2) # (bs, seq, nhead, 1, head_dim*(mem_size_factor+1))
        
        # concat      (bs,  seq,    nhead,  1,  head_dim*(mem_size_factor+1))
        # self.ffn_w1 (1,   1,      nhead,  head_dim*(mem_size_factor+1),   size_w12)                           size_w12)
        hidden = torch.matmul(concat, self.ffn_w1) # (bs, seq, nhead, 1, size_w12)
        hidden = self.ffn_activation(hidden)
        # self.ffn_w2 (1, 1, nhead, size_w12, 1)
        attn = torch.matmul(hidden, self.ffn_w2) # (bs, seq, nhead, 1, 1)
        attn = attn.squeeze(-1) # (bs, seq, nhead, 1)
        
        return attn # (bs, seq, nhead, 1)
        
        

class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim:int):
        super(VAE_Encoder, self).__init__()
        self._latent_dim = latent_dim

    def get_latent_dim(self)->int:
        return self._latent_dim

    @abstractmethod
    def pred_mu_log_var(self, batch: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        every sub class should implement this
        defines how the encoder_nn interacts with input batch in forward()
        
        this function should define EVERYTHING to be done before reparameterization
        
        should return shape (batch, input_dim of pred_mu and pred_logvar)'''
        pass

    def forward(self, batch: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        args
            batch: the main input to be encoded
            kwargs: supplementary keyword arguments, e.g. condition
        
        return
            the encoded latent vector, mu and log_var
        '''
        mu, log_var = self.pred_mu_log_var(batch, **kwargs)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor):
        # https://stats.stackexchange.com/a/16338
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(VAE_Decoder, self).__init__()
        self._latent_dim = latent_dim

    # forward function for train time
    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def infer(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def get_latent_dim(self)->int:
        return self._latent_dim
    
class TransformerUtil:
    '''
    This class is just to implement positional encoding and causal mask
    '''
    def __init__(self, d_model: int, c=10000.):
        # c is the constant when calculating positional embedding
        self.cache_seq_len = 0 # cached seq_len. to be updated
        self.dmodel = d_model
        self.c = c
        self.pe = torch.tensor(0)
        self.causal_mask = torch.tensor(0)
        self.device = torch.device('cpu')
    
    def update_pe_causal_mask(self, 
                              seq_len: int, 
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # update only when seq_len changes
        if (seq_len == self.cache_seq_len) and (self.device == device):
            return self.cache_pe, self.cache_causal_mask
        
        elif (seq_len == self.cache_seq_len) and not (self.device == device):
            self.to(device)
            return self.cache_pe, self.cache_causal_mask
        
        elif (seq_len < self.cache_seq_len):
            # if newly requested seq_len is smaller, no need to update, 
            # just return a slice of cache
            self.to(device)
            return self.cache_pe[:, :seq_len, :], self.cache_causal_mask[:seq_len, :seq_len]
        
        else:
            self.cache_causal_mask = self.calc_causal_mask(seq_len, 
                                                           device=device)
            self.cache_pe = self.calc_positional_encoding(seq_len, 
                                                          self.dmodel, 
                                                          device=device, 
                                                          c=self.c)
            self.cache_seq_len = seq_len
            self.device = device
            return self.cache_pe, self.cache_causal_mask


    @torch.no_grad
    def calc_causal_mask(self, 
                         seq_len: int, 
                         device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
    
    @torch.no_grad
    def calc_positional_encoding(self, 
                                 seq_len: int, 
                                 dmodel: int, 
                                 device:torch.device, 
                                 c=10000.) -> torch.Tensor:
        
        double_i = torch.arange(0, dmodel, 2, device=device).reshape(1, -1)
        pos = torch.arange(seq_len, device=device).reshape(-1, 1)
        odd_ = torch.sin(pos/(torch.pow(c, double_i/dmodel)))
        even_ = torch.cos(pos/(torch.pow(c, double_i/dmodel)))
        result = torch.zeros(seq_len, dmodel, device=device)
        result[:, 0::2] = odd_
        result[:, 1::2] = even_
        result = result.unsqueeze(0) # (1, seq_len, emb)
        return result

    def to(self, device: torch.device):
        self.cache_pe = self.cache_pe.to(device)
        self.cache_causal_mask = self.cache_causal_mask.to(device)
        self.device = device


class TransformerAggregator(VAE_Encoder):

    def __init__(self, 
                 latent_dim: int, 
                 encoder_layer_args: dict,
                 mem_size_factor:int=4,
                 ffn_hidden_size_factor:int=4,
                 num_stacks: int=1):
        '''
        args
            latent_dim: ...

            encoder_layer_params: params that defines an nn.TransformerEncoderLayer
            ######cheatsheet######
                d_model: int,
                nhead: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str | ((Tensor) -> Tensor) = F.relu,
                layer_norm_eps: float = 0.00001,
                batch_first: bool = False,
                norm_first: bool = False,
                bias: bool = True
            #######################
                
            num_cls_tokens: the number of class tokens as in ViT

            num_stacks: the number of stacks of TransformerDecoderLayer
        '''
        assert encoder_layer_args['d_model']%encoder_layer_args['nhead'] == 0
        assert latent_dim%encoder_layer_args['nhead'] == 0
        # for efficiency, use batch first. refer to pytorch docs
        assert 'batch_first' in encoder_layer_args.keys() and encoder_layer_args['batch_first']
        
        super().__init__(latent_dim=latent_dim)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**encoder_layer_args), num_layers=num_stacks)
        d_model = encoder_layer_args['d_model']
        
        self.util = TransformerUtil(d_model=d_model)
        
        self._d_model = d_model
        self._nhead = encoder_layer_args['nhead']
        self._head_dim = d_model//self._nhead

        self._head_ldim = latent_dim//self._nhead

        # self.agg_wk = nn.Parameter(torch.zeros(self._d_model, self._d_model), requires_grad=True)
        # nn.init.kaiming_normal_(self.agg_wk)
        
        # self.agg_wv = nn.Parameter(torch.zeros(self._d_model, self._latent_dim), requires_grad=True)
        # nn.init.kaiming_normal_(self.agg_wk)
        # self.agg_wk = nn.Parameter(torch.randn(self._d_model, self._d_model)/torch.sqrt(torch.tensor(self._d_model)), 
        #                            requires_grad=True)
        
        self.agg_wv_mean = nn.Parameter(torch.randn(self._d_model, self._latent_dim)/torch.sqrt(torch.tensor(self._d_model)), 
                                        requires_grad=True)
        
        self.agg_wv_log_var = nn.Parameter(torch.randn(self._d_model, self._latent_dim)/torch.sqrt(torch.tensor(self._d_model)), 
                                        requires_grad=True)

        self.aaa_mean = AdditiveAttentionAggregator(self._nhead, self._head_dim)
        self.aaa_log_var = AdditiveAttentionAggregator(self._nhead, self._head_dim)
        
        # self.q_mean = nn.Parameter(torch.randn(1, 1, self._nhead, self._head_dim*2), requires_grad=True)
        # self.q_log_var = nn.Parameter(torch.randn(1, 1, self._nhead, self._head_dim), requires_grad=True)
        
        self.attn_factor = 1./torch.sqrt(torch.tensor(self._head_dim))
        
    def pred_mu_log_var(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        args
            batch: input embedded sequence to encode, shape (N, seq, emb)
        '''
        
        bs, seq_len, _ = batch.shape

        pe, causal_mask = self.util.update_pe_causal_mask(seq_len, batch.device)
        batch += pe

        # encode the batch with transformer encoder
        batch = self.encoder.forward(batch, mask=causal_mask, is_causal=True) # (batch, seq, emb)
        
        # k = (batch@self.agg_wk).reshape(bs, seq_len, self._nhead, self._head_dim)
        v_mean = (batch@self.agg_wv_mean).reshape(bs, seq_len, self._nhead, self._head_ldim)
        v_log_var = (batch@self.agg_wv_log_var).reshape(bs, seq_len, self._nhead, self._head_ldim)

        # q_mean (1, 1, self._nhead, self._head_dim*n)
        # q_mean = self.q_mean.expand(bs, seq_len, -1, -1) # (bs, seq_len, nhead, hdim*n)
        # added = torch.cat([k, q_mean], dim=-1)
        # attn = (self.q_mean*k).sum(dim=-1, keepdim=True) # (bs, seq, nhead, 1)
        
        attn_mean = self.aaa_mean.forward(x=batch) # (bs, seq, nhead, 1)
        attn_mean = torch.softmax(attn_mean, dim=1) # (bs, seq, nhead, 1)
        mean = (attn_mean*v_mean).sum(dim=1) # (bs, nhead, hldm)
        mean = mean.reshape(bs, -1) # (bs, ldm)

        attn_log_var = self.aaa_log_var.forward(x=batch) # (bs, seq, nhead, 1)
        attn_log_var = torch.softmax(attn_log_var, dim=1) # (bs, seq, nhead, 1)
        log_var = (attn_log_var*v_log_var).sum(dim=1) # (bs, nhead, hldm)
        log_var = log_var.reshape(bs, -1) # (bs, ldm)

        return mean, log_var
    

class TransformerEncoder_flatten(VAE_Encoder):

    def __init__(self, 
                 latent_dim: int,
                 encoder_layer_args: dict, 
                 seq_len:int,
                 num_stacks=1):
        '''
        args
            latent_dim: ...

            encoder_layer_params: params that defines an nn.TransformerEncoderLayer
            ######cheatsheet######
                d_model: int,
                nhead: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str | ((Tensor) -> Tensor) = F.relu,
                layer_norm_eps: float = 0.00001,
                batch_first: bool = False,
                norm_first: bool = False,
                bias: bool = True
            #######################
                
            seq_len: input length. this model can only accept input of constant length

            num_stacks: the number of stacks of TransformerDecoderLayer
        '''
        # for efficiency, use batch first. refer to pytorch docs
        assert 'batch_first' in encoder_layer_args.keys() and encoder_layer_args['batch_first']
        
        super().__init__(latent_dim=latent_dim)

        d_model = encoder_layer_args['d_model']
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**encoder_layer_args), num_stacks)
        self.pred_mu = nn.Linear(seq_len*d_model, latent_dim)
        self.pred_log_var = nn.Linear(seq_len*d_model, latent_dim)

        self.seq_len = seq_len
        self.util = TransformerUtil(d_model)
        
    def pred_mu_log_var(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        args
            batch: input embedded sequence to encode, shape (N, seq, emb)
        '''
        pe, causal_mask = self.util.update_pe_causal_mask(self.seq_len, device=batch.device)
        batch += pe
        batch = self.encoder.forward(batch, mask=causal_mask, is_causal=True) # (batch, seq, emb)
        batch = batch.reshape(batch.shape[0], -1) # flatten, (N, seq_len*d_model)
        mu = self.pred_mu(batch)
        log_var = self.pred_log_var(batch)
        return mu, log_var


class TransformerEncoder_adaLN(VAE_Encoder):
    pass


class TransformerDecoder(VAE_Decoder):
    def __init__(self, 
                 latent_dim:int, 
                 decoder_layer_args:dict, 
                 num_embeddings:int,
                 num_cls_tokens:int=1,
                 num_stacks=1):
        '''
        args
            latent_dim: ...

            decoder_layer_params: params that defines an nn.TransformerEncoderLayer
            ######cheatsheet######
                d_model: int,
                nhead: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str | ((Tensor) -> Tensor) = F.relu,
                layer_norm_eps: float = 0.00001,
                batch_first: bool = False,
                norm_first: bool = False,
                bias: bool = True
            #######################
                
            num_embeddings: size of the vocabulary, output will be the estimated probabilities for each word
            
            num_cls_tokens: when expanding the latent vector, the dimension of expanded will be num_cls_tokens*d_model

            num_stacks: the number of stacks of TransformerDecoderLayer
        '''
        # for efficiency, use batch first. refer to pytorch docs
        assert 'batch_first' in decoder_layer_args.keys()
        assert 'batch_first' in decoder_layer_args and decoder_layer_args['batch_first']
        
        super().__init__(latent_dim)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**decoder_layer_args), num_stacks)
        d_model = decoder_layer_args['d_model']
        self.d_model = d_model
        self.util = TransformerUtil(d_model)
        self.num_cls_tokens = num_cls_tokens
        # self.fc_lat2emb = TwoLayerNN(latent_dim, latent_dim*2, emb_layer.embedding_dim*num_cls_tokens)
        self.fc_lat2emb = nn.Linear(latent_dim, d_model*num_cls_tokens)
        # self.fc_emb2alp = TwoLayerNN(emb_layer.embedding_dim, emb_layer.embedding_dim*2, emb_layer.num_embeddings)
        self.fc_emb2alp = nn.Linear(d_model, num_embeddings)

    # forward() for training time
    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        NOTE:
            target need not have the SOS token. 
            This function automatically pads an all-zero embedding to the target, acting as the SOS token

        args
            z: (batch, latent_dim), batch of latent vectors
            target: (batch, seq, emb) embedded target sequence

        return
            the decoded sequence of shape (batch, seq, num_embeddings),  can be used to calculate cross entropy
        '''
        # shift the target right
        target_ = torch.zeros_like(target)
        target_[:, 1:, :] = target[:, :-1, :]

        pe, causal_mask = self.util.update_pe_causal_mask(target_.shape[1], device=target.device)
        target_ += pe

        # project z to embedding dim, and make it a length-1 sequence to be the memory
        z = self.fc_lat2emb.forward(z)
        z = z.reshape(z.shape[0], self.num_cls_tokens, self.d_model) # (batch, num_cls_tokens, emb_dim)

        out = self.decoder.forward(tgt=target_, 
                                   memory=z, 
                                   tgt_mask=causal_mask, 
                                   tgt_is_causal=True)
        out = self.fc_emb2alp(out)
        return out # (batch, seq, alp_len)
    
    # infer() for inferece. Autoregressive generation
    @torch.no_grad
    def infer(self, 
              z: torch.Tensor, 
              max_len: int, 
              embedding_layer: nn.Embedding,
              padding_idx:int=0,
              stop_on_pad:bool=False,
              temperature:float=0.0) -> torch.Tensor:
        '''
        This function is decorated by torch.no_grad, do not call in training loop.

        args
            z (batch, latent_dim), batch of latent vectors

            padding_idx: the index of [PAD] token

            max_len: int, maximum length of generation

            embedding_layer: ...

            stop_on_pad: when set to False, will decode until max_len reached. Renders padding_idx ineffective.
            [TIP] During inference can be set to True, thus when all tokens are predicted to be [PAD], the generation stops
            
            temperature: if set to 0.0, uses greedy decoding.

        return
            decoded index vector Tensor of shape (batch, seq_len)
                seq_len is always max_len when stop_on_pad=True

            NOTE: decoded index vector will not have a SOS token at front.
        '''
        assert temperature >= 0.0
        if temperature < 1e-6:
            use_greedy = True
        else:
            use_greedy = False
            
        bs, _ = z.shape
        
        z = self.fc_lat2emb.forward(z)
        z = z.reshape(bs, self.num_cls_tokens, self.d_model)

        out = torch.zeros(bs, max_len, dtype=torch.long, device=z.device)
        out_emb = torch.zeros(bs, max_len+1, self.d_model, device=z.device) # (batch, max_len+1, emb)
        
        for i in range(max_len):
            current_emb = out_emb[:, 0:i+1, :].clone()
            pe, causal_mask = self.util.update_pe_causal_mask(i+1, device=z.device)
            current_emb += pe
            pred = self.decoder.forward(current_emb, z, tgt_mask=causal_mask, tgt_is_causal=True) # (batch, i+1, emb)
            # get the last predicted token
            pred = pred[:, -1, :] # (batch, emb)
            pred = self.fc_emb2alp(pred) # (batch, alp)
            
            if use_greedy:
                # get the most probable index
                next_idx = torch.argmax(pred, dim=1) # (batch, )
            else:
                probs = torch.softmax(pred/temperature, dim=1)
                next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            if stop_on_pad and torch.all(next_idx==padding_idx): # if all are finished, return
                return out[:, :i]
            
            # update the output indices
            out[:, i] = next_idx
            # update the embeddings
            out_emb[:, i+1, :] = embedding_layer.forward(next_idx) # (batch, emb)
        
        return out # (batch, seq_len)
    

ENCODER_TYPE = {'transformer_encoder_flatten': TransformerEncoder_flatten,
                'transformer_aggregator': TransformerAggregator}

DECODER_TYPE = {'transformer_decoder': TransformerDecoder}


class TransformerVAE(nn.Module):

    def __init__(self, 
                 embedding_layer: nn.Embedding, 
                 encoder: VAE_Encoder, 
                 decoder: TransformerDecoder,
                 latent_regressor: nn.Module,
                 weight_tying:bool=False):
        '''
        Args
            ...
            weight_tying: use shared weights in the embedding layer and the final projection layer.
                refer to https://arxiv.org/abs/1611.01462
        '''
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.decoder = decoder

        self.weight_tying = weight_tying
        if weight_tying:
            # do weight tying
            self.embedding_weight = nn.Parameter(torch.zeros_like(self.embedding_layer.weight))
            nn.init.kaiming_normal_(self.embedding_weight)
            self.embedding_layer.weight = self.embedding_weight
            self.decoder.fc_emb2alp.weight = self.embedding_weight

        self.latent_regressor = latent_regressor
        self.latent_dim = self.encoder.get_latent_dim()

    def forward(self, 
                x:torch.Tensor,
                encoder_model_kwargs:dict={},
                decoder_model_kwargs:dict={},
                regressor_model_kwargs:dict={}) -> tuple[torch.Tensor, ...]:
        '''
        Intended for calculating elbo and latent regressor loss during training
        args
            batch: input idxv, shape (batch, seq)
            NOTE: samples need not have SOS token

        return
            latent vector z, 
            predicted mu, 
            predicted log_var, 
            reconstructed input of shape (batch, seq, num_embeddings)
            predicted property
        '''
        embeddings = self.embedding_layer.forward(x) # (batch, seq, emb)
        z, mu, log_var = self.encoder.forward(embeddings, **encoder_model_kwargs)
        
        embeddings = self.embedding_layer.forward(x) # something magical... this accelerates the training progress
        # it could be that this step reduces gradient vainishing
        x_hat = self.decoder.forward(z, embeddings, **decoder_model_kwargs) # (batch, seq, num_embeddings)
        
        if self.latent_regressor is None:
            return z, mu, log_var, x_hat, None
        
        pred_prop = self.latent_regressor(z, **regressor_model_kwargs)
        return z, mu, log_var, x_hat, pred_prop
    
    def encode(self, x: torch.Tensor, model_kwargs:dict={}) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Encode the input index vectors of shape (batch, seq)

        Args
            x: input padded index vectors
            model_kwargs: the additional arguments to pass into encoder.forward. CVAE may need this.

        Returns
            z, mu, log_var
        
        '''
        embeddings = self.embedding_layer.forward(x) # (batch, seq, emb)
        return self.encoder.forward(embeddings, **model_kwargs)
    
    def decode(self, 
               z:torch.Tensor, 
               target:torch.Tensor, 
               model_kwargs:dict={}) -> torch.Tensor:
        '''
        Teacher-forced decoding, requires target.
        NOTE: target need not contain preceding SOS tokens
        '''
        return self.decoder.forward(z, target, **model_kwargs)
    
    def decode_autoregressive(self,
                              z: torch.Tensor, 
                              max_len: int,
                              padding_idx:int=0,
                              stop_on_pad:bool=False,
                              temperature:float=0.0,
                              model_kwargs:dict={}) -> torch.Tensor:
        '''
        Auto-regressive decoding from latent vector z. 
        
        Won't keep computational graph, do not call in training loop.
        
        Args&Return: same with TransformerDecoder.infer
        '''
        return self.decoder.infer(z=z, 
                                  max_len=max_len,
                                  embedding_layer=self.embedding_layer,
                                  padding_idx=padding_idx,
                                  stop_on_pad=stop_on_pad,
                                  temperature=temperature,
                                  **model_kwargs)
    
    def training_loss(self, 
                      x: torch.Tensor,
                      kld_beta:float,
                      y: torch.Tensor|None=None,
                      prop_loss_alpha:float|None=None,
                      l1_reg_lambda:float|None=None,
                      group_dim:int=1,
                      encoder_model_kwargs:dict={},
                      decoder_model_kwargs:dict={},
                      regressor_model_kwargs:dict={}) -> torch.Tensor:
        '''
        Args
            x: input index vectors of shape (batch, seq)
            NOTE: x need not have preceding SOS tokens
            NOTE: if use l1_reg, the latent_regressor has to be an instance of nn.Linear
            
            y: properties, set to None if no need
            
            kld_beta: the beta as in beta-VAE, the factor to be multiplied to KL-divergence term
            
            prop_loss_alpha: the factor to be multiplied to the mse_loss of property regressor

            l1_reg_lambda: the coefficient of l1 regularization penalty on the latent regressor, set to None if not needed

            group_dim: num of dimensions per group, 
                when set to 1, group LASSO degrades to vanilla LASSO
                when set equal to the latent_dim, group LASSO degrades to L2 regularization

        Returns
            the loss tensor
        '''
        z, mu, log_var, x_hat, pred_prop = self.forward(x=x,
                                                        encoder_model_kwargs=encoder_model_kwargs,
                                                        decoder_model_kwargs=decoder_model_kwargs,
                                                        regressor_model_kwargs=regressor_model_kwargs)

        l_kld = -0.5 * torch.mean(1. + log_var - mu.pow(2) - log_var.exp())

        l_recon = cross_entropy(x_hat.reshape(-1, x_hat.shape[-1]), x.reshape(-1))

        if y is None:
            # even if has regressor, can skip training it when y is None
            return kld_beta*l_kld + l_recon
        
        # now we have y, so we must have regressor
        if self.latent_regressor is None:
            raise ValueError('TransformerVAE initialized without latent regressor but got provided y')
        # also, we must have the coefficient for regressor loss
        if prop_loss_alpha is None:
            raise ValueError('If regressor is to be trained, prop_loss_alpha must be provided')

        l_regr = nn.functional.mse_loss(pred_prop, y)
        if l1_reg_lambda is None:
            return prop_loss_alpha*l_regr + kld_beta*l_kld + l_recon        
    
        l_regr += l1_reg_lambda*reg_group_lasso(self.latent_regressor, group_splits=group_dim)

        return prop_loss_alpha*l_regr + kld_beta*l_kld + l_recon


# define the regularization penalty functions here
# def reg_lasso(model: nn.Module) -> torch.Tensor:
#     norm_list = []
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             norm_list.append(torch.norm(param, p=1, keepdim=True))
#     return torch.sum(torch.cat(norm_list))


def reg_group_lasso(model: nn.Linear, group_splits: list[int] | int) -> torch.Tensor:
    '''
    args
        model: the linear layer to be regularized

        group_splits: sizes of groups (when use int, try to split into equal sized groups)
            Example:
                weight size (n, 10) to split two equal groups, group_splits = [5, 5]
    '''
    result = torch.tensor(0., device=model.weight.device, dtype=model.weight.dtype)
    for g in torch.split(model.weight, group_splits, dim=1):
        factor = torch.sqrt(torch.tensor(g.shape[1], device=model.weight.device))
        result += factor * torch.sum(torch.norm(g, p=2, dim=1))
    return result


def model_config_parser(config: dict) -> TransformerVAE:
    encoder_config = config['encoder']
    encoder_cls = ENCODER_TYPE[encoder_config['backbone']]
    encoder_args = encoder_config['model_args']
    encoder = encoder_cls(**encoder_args)

    decoder_config = config['decoder']
    decoder_cls = DECODER_TYPE[decoder_config['backbone']]
    decoder_args = decoder_config['model_args']
    decoder = decoder_cls(**decoder_args)

    regressor = nn.Linear(config['latent_dim'], config['num_properties'])
    embedding_layer = nn.Embedding(**config['embedding_layer_args'])

    vae_kwargs = config.get('vae_kwargs', {})
    return TransformerVAE(embedding_layer=embedding_layer,
                          encoder=encoder,
                          decoder=decoder,
                          latent_regressor=regressor,
                          **vae_kwargs)


# unit testing: overfitting a dummy batch
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import os
    pwd = os.path.dirname(__file__)
    paths = OmegaConf.load(f'{pwd}/path.yml')
    path_to_model_config = paths['path_to_model_config']
    # model_config = OmegaConf.load(os.path.join(path_to_model_config, 'tvae_agg.tiny8_4.yml'))
    
    num_samples = 24
    seq_len = 10
    num_embeddings = 8
    dummy_batch = torch.randint(low=0, high=8, size=(num_samples, seq_len), dtype=torch.long)
    
    model_config = '''
    latent_dim: 4
    embedding_layer_args:
        embedding_dim: 8
        num_embeddings: 17
        padding_idx: 0

    encoder:
        backbone: 'transformer_aggregator'
        model_args:
            latent_dim: ${latent_dim}
            num_stacks: 4
            mem_size_factor: 1
            ffn_hidden_size_factor: 2
            encoder_layer_args:
                d_model: ${embedding_layer_args.embedding_dim}
                dim_feedforward: 256
                nhead: 4
                batch_first: True

    num_properties: 5

    decoder:
        backbone: 'transformer_decoder'
        model_args:
            num_stacks: 4
            decoder_layer_args:
                dim_feedforward: 256
                nhead: 4
                batch_first: True
                d_model: ${embedding_layer_args.embedding_dim}
            
            num_cls_tokens: 32
            num_embeddings: ${embedding_layer_args.num_embeddings}
            latent_dim: ${latent_dim}

    vae_kwargs:
        weight_tying: True
        '''
    model_config = OmegaConf.create(model_config)
    vae = model_config_parser(model_config)
    print(vae)
    
    
    from torch.optim import Adam
    opt = Adam(vae.parameters(), 1e-3)
    vae.train()
    num_epochs = 2000
    for i in range(num_epochs):
        loss = vae.training_loss(dummy_batch, kld_beta=1e-5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i+1)%(num_epochs//10) == 0:
            print(f'loss at iter {i:d}: {loss.item():.4f}')
    
    vae.eval()
    # reconstruct
    z, _, _ = vae.encode(dummy_batch)
    recon = vae.decode_autoregressive(z, max_len=seq_len)
    
    quality = torch.mean((dummy_batch==recon).float())
    print(f'recon. quality: {quality.item():.4f}')