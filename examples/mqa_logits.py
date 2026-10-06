"""MXFP8 MQA logits (lightning indexer) example.

build https://github.com/deepseek-ai/DeepGEMM before running this example.

Run from the repository root:
    python examples/mqa_logits.py
    binet ls -- python examples/mqa_logits.py
"""

import torch

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8

SQ, SKV, HEADS, DIM = 4096, 8192, 64, 128

torch.manual_seed(0)
q = torch.randn(SQ, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(SKV, DIM, device="cuda", dtype=torch.bfloat16)
weights = torch.randn(SQ, HEADS, device="cuda", dtype=torch.bfloat16)  # rows are 16-byte aligned at 64 heads


def mxfp8(x, *shape):
    """One UE8M0 scale per 32 elements, packed 4 scales per int32 (SM100 MXFP8)."""
    fp8, scale = per_token_cast_to_fp8(x.view(-1, DIM), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    return fp8.view(*shape), scale.view(shape[:-1])


ks = torch.zeros(SQ, dtype=torch.int, device="cuda")
ke = torch.full((SQ,), SKV, dtype=torch.int, device="cuda")  # non-causal: every query sees all KV

logits = deep_gemm.fp8_fp4_mqa_logits(q=mxfp8(q, SQ, HEADS, DIM), kv=mxfp8(kv, SKV, DIM), weights=weights,
                                      cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=SKV)

torch.cuda.synchronize()
print(f"{logits.dtype} {tuple(logits.shape)} sum={logits.float().sum().item():.1f}", flush=True)
