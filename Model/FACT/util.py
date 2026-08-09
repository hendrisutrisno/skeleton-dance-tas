import torch
import torch.nn.functional as F
import numpy as np
import os
from yacs.config import CfgNode
from FACT.default import get_cfg_defaults

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1
    
    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)
    
    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e-s+1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e-s+1


def parse_label(label: np.array):
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    loc = label[:-1] != label[1:]
    loc = np.where(loc)[0]
    segs = []
    
    if len(loc) == 0:
        return [ Segment(label[0], 0, len(label)-1) ]
        
    for i, l in enumerate(loc):
        if i == 0:
            start = 0
            end = l
        else:
            start = loc[i-1]+1
            end = l
        
        seg = Segment(label[start], start, end)
        segs.append(seg)
        
    segs.append(Segment(label[loc[-1]+1], loc[-1]+1, len(label)-1))
    return segs

def setup_cfg(cfg_file=[], set_cfgs=None, default = None, logdir="log/"):
    """
    update default cfg according to cmd line input
    and automatic generate experiment name
    """
    cfg = get_cfg_defaults()

    # preprocess set_cfgs to convert int2float
    L = len(set_cfgs) if set_cfgs else 0
    new_set_cfgs = []
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if not isinstance(k, list):
                k = [k]
        for k_ in k:
            tgt = _get_var(cfg, k_.split('.'))
            v_ = int2float_check(v, tgt)
            new_set_cfgs.extend([k_, v_])


    # update cfg
    for f in cfg_file: # if no config file, this is empty list
        cfg.merge_from_file(f)
    if set_cfgs is not None:
        cfg.merge_from_list(new_set_cfgs)
    cfg.aux.cfg_file = cfg_file
    cfg.aux.set_cfgs = set_cfgs

    # generate experiment name
    cfg.aux.exp = generate_expname(cfg, default=default)

    logdir = logdir if not cfg.aux.debug else "log_test/"
    logdir = os.path.join(logdir, cfg.dataset, cfg.split,
                                    cfg.aux.exp, str(cfg.aux.runid))
    logdir = logdir.replace('-', '_') 

    cfg.aux.logdir = logdir
    return cfg

def _get_var(c, ks: list, delete=False):
    if len(ks) == 1:
        v = c[ks[0]]
        if delete:
            del c[ks[0]]
        return v
    else:
        return _get_var(c[ks[0]], ks[1:], delete=delete)
    
def int2float_check(x, tgt):
    if isinstance(tgt, float) and "." not in x:
        try:
            int(x) # first check if x can convert to int
            x = x + '.0' # if can convert, change to float match str
        except ValueError:
            pass # cannot convert, pass on to cfg to throw error
    return x  
_CONFIG_FILE_DICT = {}
def generate_expname(cfg, cfg_file=None, default=None) -> str:
    if cfg_file is None:
        cfg_file = cfg.aux.cfg_file

    expname = []
    default = get_cfg_defaults()

    for f in cfg_file:
        if f not in _CONFIG_FILE_DICT:
            with open(f, 'r') as fp:
                _CONFIG_FILE_DICT[f] = CfgNode.load_cfg(fp)

        default.merge_from_other_cfg(_CONFIG_FILE_DICT[f])

        f = os.path.basename(f)
        f = '.'.join(f.split('.')[:-1])
        expname.append(f)


    # add other setting
    diff = generate_diff_dict(default, cfg)
    prune = {}
    for k, v in diff.items():
        prune[capitalize(k)] = v
    diff_string = diff2expname(prune)
    if len(diff_string) > 0:
        expname.append(diff_string)
    if len(cfg.aux.mark) > 0:
        expname.append(cfg.aux.mark)

    expname = '-'.join(expname)
    return expname

def generate_diff_dict(default: CfgNode, cfg: CfgNode, include_missing=False) -> dict :
    """
    include_missing = False
        if a key is missing in cfg,
        it assumes the value matches with that of default
    """

    diff = {}
    for k, v in cfg.items():
        if k not in default and (not include_missing):
            continue
        if isinstance(v, CfgNode):
            subdiff = generate_diff_dict(default[k], cfg[k], include_missing=include_missing)
            if len(subdiff) > 0:
                diff[k] = subdiff
        else:
            if v != default[k]:
                diff[k] = v
    
    return diff

def capitalize(string):
    return string[0].upper() + string[1:]

def diff2expname(diff: dict, remove_leaf=False):
    string = ""
    for k, v in diff.items():
        if k.lower()  == "aux":
            continue # exclude auxiliary config
        if k.lower() == "split":
            continue # exclude split name

        if isinstance(v, dict):
            v = diff2expname(v, remove_leaf=False) # when recursive call, always false
            string += "%s[%s]-" % (k, v)
        elif not remove_leaf:
            if isinstance(v, bool):
                v = str(v)[0]
            string += "%s:%s-" % (k, v)
    
    string = string[:-1] # remove last dash
    return string