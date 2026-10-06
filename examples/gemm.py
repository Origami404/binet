"""A single bf16 x bf16 GEMM of 4096^3, launched through PyTorch."""

# Run from the repository root:
#   python examples/gemm.py
#   binet ls -- python examples/gemm.py
#
# Replace KERNEL_NAME with the GEMM kernel name printed by ls:
#   binet get --kernel-name "KERNEL_NAME" --output gemm.cubin -- python examples/gemm.py
#
# Choose one or more probeable sites from gemm.info.json.
# Replace 12 48 77 below with the selected integer site indices:
#   binet mv gemm.cubin --kernel-name "KERNEL_NAME" --sites 12 48 77 --output prepared.cubin
#
# Capture the first matching launch:
#   binet profile --cubin prepared.cubin --output gemm-profile -- python examples/gemm.py
# Read gemm-profile/trace.npz with binet.trace.Trace or numpy.load.

import torch

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

c = a @ b

torch.cuda.synchronize()
print(f"{c.dtype} {tuple(c.shape)} sum={c.sum().item():.1f}", flush=True)
