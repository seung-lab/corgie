import torch
import torch.nn as nn

from blockmatch import block_match


class SeethroughBlack(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_img, tgt_img, **kwargs):
        return ((src_img == 0) * (tgt_img != 0)).float()



def create(**kwargs):
    return SeethroughBlack(**kwargs)
