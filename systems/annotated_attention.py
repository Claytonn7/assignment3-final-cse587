import math
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
from basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    with nvtx.range("computing attention scores"):
        S = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(K.shape[-1])
        if mask is not None:
            S = torch.where(mask, S, float("-inf"))
    with nvtx.range("computing softmax"):
        P = softmax(S, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(P, V, "... q k, ... k d -> ... q d")
