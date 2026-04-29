import math
import torch 
import torch.nn.functional as F

from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        # Use PyTorch's native memory-efficient attention (FlashAttention)
        # which dramatically prevents OOM by avoiding N^2 matrix allocation
        if mask is not None:
            # Convert the numerical 0/1 mask into a boolean mask for the new API
            bool_mask = (mask != 0)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
        else:
            output = F.scaled_dot_product_attention(q, k, v)
        return output
