import torch
import torch.nn as nn
import math
from FACT.loss import smooth_loss
from Ms_tcn import stage_plus
from FACT.util import _get_activation_fn, parse_label
import scipy.ndimage
import copy
import numpy as np

def logit2prob(clogit, dim=-1, class_sep=None):
    if class_sep is None or class_sep<=0:
        cprob = torch.softmax(clogit, dim=dim)
    else:
        assert dim==-1, dim
        cprob1 = torch.softmax(clogit[..., :class_sep], dim=dim)
        cprob2 = torch.softmax(clogit[..., class_sep:], dim=dim)
        cprob = torch.cat([cprob1, cprob2], dim=dim)
    
    return cprob

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=5000, empty=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.empty = empty
        self.__compute_pe__(d_model, max_len)


    def __compute_pe__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        if not self.empty:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)
    
    def __str__(self):
        if self.empty:
            return f"PositionalEncoding(EMPTY)"
        else:
            return f"PositionalEncoding(Dim={self.d_model}, MaxLen={self.max_len})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x.dim0 = sequence length
            output: [sequence length, batch_size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if x.size(0) > self.pe.shape[0]: 
            self.__compute_pe__(self.d_model, x.size(0)+10)
            self.pe = self.pe.to(x.device)

        return self.pe[:x.size(0), :]

def add_positional_encoding(tensor, pos):
    if pos is None:
        return tensor
    else:
        d = pos.size(-1)
        tensor = tensor.clone()
        tensor[:, :, :d] = tensor[:, :, :d] + pos
        return tensor

class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super(Block, self).__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of action classes
        clogit = feature[:, :, -nclass:] # class logit
        feature = feature[:, :, :-nclass] # feature without clogit
        cprob = logit2prob(clogit, dim=-1)  # apply softmax
        feature = torch.cat([feature, cprob], dim=-1)

        return feature, clogit

    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim
        frame_branch = stage_plus(cfg.f_layers, cfg.f_dim, in_dim, cfg.hid_dim, cfg.dropout)

        return frame_branch

    def create_abranch(self, cfg):
        if cfg.a == 'sa': # self-attention layers, for update blocks
            l = SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca': # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        else:
            raise ValueError(cfg.a)

        return action_branch

    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, 
            head_dim=cfg.hid_dim,
            dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer

    @staticmethod
    def _eval(action_clogit, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1) #(L, 1, F)

        action_clogit = action_clogit.squeeze(1) #(M, F+1)
        a2f_attn = a2f_attn.squeeze(0) # L, M
        qtk_cpred = action_clogit.argmax(1)
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob, fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1) # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob, prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript] 

        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N] # 1, f, a -> f, s'
        abranch_prob = torch.softmax(a2f_attn, dim=-1) # f, s'

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1) # f
        pred = transcript[pred]
        return prob, pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)

class InputBlock(Block):
    def __init__(self, cfg, in_dim, nclass):
        super(InputBlock, self).__init__()
        self.cfg = cfg
        self.nclass = nclass
        cfg = cfg.Bi
        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos, action_clogit=None):
        frame_feature = frame_feature.permute([1, 2, 0])  #(T, 1, H) to (1, H, T)
        frame_feature = self.frame_branch(frame_feature)  #(1, H, T)
        frame_feature = frame_feature.permute([2, 0, 1])  #(1, H, T) to (T, 1, H)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # action branch
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)
        
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit

        return frame_feature, action_feature

    def compute_loss(self, criterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smoothloss = smooth_loss(frame_clogit)

        return frame_loss + atk_loss + self.cfg.Loss.sw * smoothloss
    
class UpdateBlock(Block):
    def __init__(self, cfg, nclass):
        super(UpdateBlock, self).__init__()
        self.cfg = cfg
        self.nclass = nclass
        cfg = cfg.Bu
        self.frame_branch = self.create_fbranch(cfg)
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)
        self.action_branch = self.create_abranch(cfg)
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)
        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = frame_feature.permute([1, 2, 0])  #(T, 1, H) to (1, H, T)
        frame_feature = self.frame_branch(frame_feature)  #(1, H, T)
        frame_feature = frame_feature.permute([2, 0, 1])  #(1, H, T) to (T, 1, H)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit 
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, action_feature

    def compute_loss(self, criterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1)) 
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = smooth_loss( self.a2f_attn_logit )
        fl = smooth_loss( torch.transpose(self.f2a_attn_logit, 1, 2) )
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) # f, 1, c -> 1, f, c
        l = smooth_loss( frame_clogit )
        smoothloss = al + fl + l

        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.cfg.Loss.sw * smoothloss

class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, nclass):
        super(UpdateBlockTDU, self).__init__()
        self.cfg = cfg
        self.nclass = nclass
        cfg = cfg.BU

        self.frame_branch = self.create_fbranch(cfg)
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim//2, cfg.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)
        self.action_branch = self.create_abranch(cfg)
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim), nn.ReLU())

    def temporal_downsample(self, frame_feature):

        # get action segments based on predictions
        cprob = frame_feature[:, :, -self.nclass:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = pred.detach().cpu().numpy()
        segs = parse_label(pred)
        tdu = TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H

        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = frame_feature.permute([1, 2, 0])  #(T, 1, H) to (1, H, T)
        frame_feature = self.frame_branch(frame_feature)  #(1, H, T)
        frame_feature = frame_feature.permute([2, 0, 1])  #(1, H, T) to (T, 1, H)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation       
        self.frame_clogit = frame_clogit
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.action_clogit = action_clogit 

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) 
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smoothloss = smooth_loss( frame_clogit )

        return (frame_loss + seg_loss)/ 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * smoothloss
    
class SALayer(nn.Module):
    """
    self or cross attention
    """

    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 activation="relu", vpos=False):
        super(SALayer, self).__init__()

        kv_dim = q_dim if kv_dim is None else kv_dim
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim=kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.use_vpos = vpos
        self.dropout_rate = (dropout, attn_dropout)

    def __str__(self) -> str:
        return f"SALayer( q({self.q_dim})xkv({self.kv_dim})->{self.q_dim}, head:{self.nhead}, ffdim:{self.dim_feedforward}, dropout:{self.dropout_rate}, vpos:{self.use_vpos} )"
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, key, value, 
            query_pos = None,
            key_pos = None,
            value_pos = None):
        """
        tgt : query
        memory: key and value
        """
        query = add_positional_encoding(tgt, query_pos)
        key = add_positional_encoding(key, key_pos)
        if self.use_vpos:
            value=add_positional_encoding(value, value_pos)

        tgt2, self.attn = self.multihead_attn(query, key, value, average_attn_weights=False) # attn: nhead, batch, q, k

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt 
    
class SCALayer(nn.Module):
    def __init__(self, action_dim, frame_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1,
                 activation="relu", normalize_before=False, 
                 sa_value_w_pos=False, ca_value_w_pos=False):
        """
        Self-Attention + Cross-Attention Module
        """
        super(SCALayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(action_dim, nhead, dropout=attn_dropout)
        self.multihead_attn = nn.MultiheadAttention(action_dim, nhead, kdim=frame_dim, vdim=frame_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(action_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, action_dim)

        self.norm1 = nn.LayerNorm(action_dim)
        self.norm2 = nn.LayerNorm(action_dim)
        self.norm3 = nn.LayerNorm(action_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        assert not normalize_before
        self.sa_value_w_pos = sa_value_w_pos
        self.ca_value_w_pos = ca_value_w_pos
        
        
        self.string = f"SCALayer( adim:{action_dim}, fdim:{frame_dim}, head:{nhead}, ffdim:{dim_feedforward}, dropout:{(dropout, attn_dropout)}, svpos:{sa_value_w_pos}, cvpos:{ca_value_w_pos} )"

    def __str__(self) -> str:
        return self.string
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, memory,
                     pos = None,
                     query_pos = None):
        # self attention
        q = k = add_positional_encoding(tgt, query_pos)
        if not self.sa_value_w_pos:
            tgt2, self.sa_attn = self.self_attn(q, k, tgt)
        else:
            tgt2, self.sa_attn = self.self_attn(q, k, q)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        query=add_positional_encoding(tgt, query_pos)
        value = memory
        key = add_positional_encoding(memory, pos)

        if not self.ca_value_w_pos:
            tgt2, self.ca_attn = self.multihead_attn(query, key, value)
        else:
            tgt2, self.ca_attn = self.multihead_attn(query, key, key)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class SCADecoder(nn.Module):
    """
    Self+Cross-Attention Decoder
    """

    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer, num_layers, norm=None, in_map=False):
        super(SCADecoder, self).__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert hid_dim == in_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, pos = None, query_pos = None):

        if self.in_map:
            output = self.in_linear(tgt)
        else:
            output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        output = self.out_linear(output)
        return output

class SADecoder(nn.Module):
    """
    Self-Attention Decoder
    """

    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer: SALayer, num_layers, norm=None, in_map=False):
        super(SADecoder, self).__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert in_dim == hid_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, pos = None):

        if self.in_map:
            output = self.in_linear(tgt)
        else:
            output = tgt

        for layer in self.layers:
            output = layer(output, output, output, query_pos=pos, key_pos=pos, value_pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        output = self.out_linear(output)

        return output

class X2Y_map(nn.Module):

    def __init__(self, x_dim, y_dim, y_outdim, head_dim, dropout=0.5, kq_pos=False):
        super(X2Y_map, self).__init__()
        self.kq_pos = kq_pos

        self.X_K = nn.Linear(x_dim, head_dim)
        self.X_V = nn.Linear(x_dim, head_dim)
        self.Y_Q = nn.Linear(y_dim, head_dim)

        self.Y_W = nn.Linear(y_dim+head_dim, y_outdim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X_feature, Y_feature, X_pos=None, Y_pos=None, X_pad_mask=None, Y_pad_mask=None):
        """
        X: x, b, h
        Y: y, b, h
        """
        X = X_feature.shape[0]
        Y = Y_feature.shape[0]

        if (X_pos is not None) and self.kq_pos:
            x = add_positional_encoding(X_feature, X_pos)
            xk = self.X_K(x) 
        else:
            xk = self.X_K(X_feature) 

        xv = self.X_V(X_feature)

        if (Y_pos is not None) and self.kq_pos:
            y = add_positional_encoding(Y_feature, Y_pos)
            yq = self.Y_Q(y)
        else:
            yq = self.Y_Q(Y_feature)

        assert X_pad_mask is None and Y_pad_mask is None

        attn_logit = torch.einsum('xbd,ybd->byx', xk, yq)
        attn_logit = attn_logit / math.sqrt(xk.shape[-1])
        self.attn_logit = attn_logit
        attn = torch.softmax(attn_logit, dim=-1) # B, y, x
        # if self.drop_on_att:
        #     attn = self.dropout(attn)
        
        attn_feat = torch.einsum('byx,xbh->ybh', attn, xv)
        concat_feature = torch.cat([Y_feature, attn_feat], dim=-1)
        concat_feature = self.dropout(concat_feature)
        # if not self.drop_on_att:

        Y_feature = self.Y_W(concat_feature)

        self.attn = attn.unsqueeze(1) # B, nhead=1, X, Y

        return Y_feature

class TemporalDownsampleUpsample():

    def __init__(self, segs):
        self.segs = segs
        self.num_seg = len(segs)

        self.seg_label = []
        for i, seg in enumerate(segs):
            self.seg_label.extend([i]*seg.len)
        self.seg_label = torch.LongTensor(self.seg_label) #.cuda()
        self.seg_lens = torch.LongTensor([s.len for s in segs]) #.cuda()

    def cuda(self):
        self.seg_label = self.seg_label.cuda()
        self.seg_lens = self.seg_lens.cuda()

    def to(self, device):
        self.seg_label = self.seg_label.to(device)
        self.seg_lens = self.seg_lens.to(device)

    def feature_frame2seg(self, frame_feature, normalize=True):
        f, b, h = frame_feature.shape
        assert b == 1
        seg_feature = torch.zeros(self.num_seg, b, h, device=frame_feature.device)
        seg_feature.index_add_(0, self.seg_label, frame_feature)

        if normalize:
            seg_feature = seg_feature / self.seg_lens[:, None, None]

        return seg_feature

    def attn_frame2seg(self, frame_attn):
        b, f, a = frame_attn.shape
        assert b == 1

        seg_attn = torch.zeros(b, self.num_seg, a, device=frame_attn.device)
        seg_attn.index_add_(1, self.seg_label, frame_attn)

        seg_attn = seg_attn / self.seg_lens[:, None]

        return seg_attn

    def feature_seg2frame(self, seg_feature):
        """
        seg_feature : S, B, H
        """
        frame_feature = seg_feature[self.seg_label]
        return frame_feature

    def attn_seg2frame(self, seg_attn):
        """
        seg_attn : B, S, A
        """
        assert seg_attn.shape[0] == 1
        frame_attn = seg_attn[0, self.seg_label].unsqueeze(0)
        return frame_attn