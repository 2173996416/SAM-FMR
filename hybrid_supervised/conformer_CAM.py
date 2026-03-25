import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.set_printoptions(threshold=np.inf)

from hybrid_supervised import conformer_sam_fmr
from Conformer import conformer_weak


class Net_sam_fmr(conformer_sam_fmr.Net):
    def __init__(self):
        super(Net_sam_fmr, self).__init__(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=2)  # todo
