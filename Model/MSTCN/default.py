from yacs.config import CfgNode as CN

_C = CN()

# ============================================================
# Model
# ============================================================

_C.model = CN()
_C.model.num_blocks = 4
_C.model.num_layers = 10
_C.model.num_f_maps = 128
_C.model.num_classes = 5
_C.model.channel_mask_rate = 0.3


_C.dataset = CN()
_C.dataset.feature_dim = 50
_C.dataset.sample_rate = 1
_C.dataset.batch_size = 1
_C.dataset.lr = 0.0001
_C.dataset.epochs = 300

# ============================================================
# Loss
# ============================================================

_C.loss = CN()
_C.loss.ignore_index = -100


def get_cfg_defaults():
    """
    Get a copy of the default config.
    """
    return _C.clone()
