# VRAM benchmark

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB |
| CPU | Intel i5-12500H |
| RAM | 16 GB |
| OS | Windows 11 (26200) |
| Driver | 591.74 |
| PyTorch | 2.6.0 + cu124 |
| diffusers | 0.42.1 |
| dtype | bfloat16 |
| LoRA | rank 16, targets to_q / to_k / to_v / to_out.0 |
| Gradient checkpointing | on |

## Method

```bash
python benchmark.py \
    --base-model /path/to/sdxl.safetensors \
    --data ./dataset/karisa_v3 \
    --resolutions 512 --images 1 --steps 10
```

Peak VRAM is BTtorch.cuda.max_memory_allocated()BT, reset immediately before the
training loop starts, so it excludes the one-off cost of loading the checkpoint.

## Results

| Resolution | Peak VRAM | Wall time per step | Note |
|---|---|---|---|
| 512 x 512 | **5.20 GB** | 11.0 s/it over the first 10 steps | includes CUDA warm-up |
| 512 x 512 | 5.2 GB | **~1.0 s/it steady state** | measured over a 760-step run |
| 768 x 768 | 6.1 GB | ~2.6 s/it | close to the 6 GB ceiling |
| 1024 x 1024 | - | - | out of memory |

## Why the first steps are slow

CUDA lazily compiles and autotunes kernels for the shapes it meets. In a 10-step
benchmark that warm-up is a large share of the wall time, so short runs report
inflated seconds-per-iteration. Steady-state numbers above come from full runs
(38 images x 4-5 epochs) where the first tens of steps are a rounding error.

Reporting both numbers is deliberate: the short-run figure is what the script
actually prints, and pretending otherwise would make the repo's numbers
irreproducible.

## What limits you

At 512 px the bottleneck is the UNet forward/backward pass, not the optimiser.
Reducing rank from 16 to 8 saves almost no VRAM (the adapter is tiny); the memory
goes to activations, which is why gradient checkpointing does the heavy lifting.
