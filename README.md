# Binet

Binary-level event tracing for CUDA kernels.

## Installation

```bash
python -m pip install git+https://github.com/Chtholly-Boss/binet.git
```

## Usage

```bash
binet ls -- python workload.py

binet get --kernel-name my_kernel --output kernel.cubin --num-threads 256 -- python workload.py

binet mv kernel.cubin --kernel-name my_kernel --sites 12 48 77 --output prepared.cubin

binet profile --cubin prepared.cubin --output run-001 -- python workload.py
```
