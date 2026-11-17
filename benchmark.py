#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure peak VRAM and step time for a few resolutions.

Usage:
    python benchmark.py --base-model path/to/sdxl.safetensors --data ./dataset/x
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_cache
from src.trainer import LowVramTrainer, peak_vram_gb, reset_peak_vram
from train import parse_args as _unused  # noqa: F401  (keeps imports explicit)


def run(args, resolution):
    from diffusers import StableDiffusionXLPipeline

    torch.cuda.empty_cache()
    reset_peak_vram()

    pipe = StableDiffusionXLPipeline.from_single_file(
        args.base_model, torch_dtype=torch.bfloat16, use_safetensors=True
    )
    cfg = argparse.Namespace(
        rank=args.rank, alpha=args.rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        resolution=resolution, lr=args.lr, epochs=1, steps_per_image=args.steps,
        seed=42, dtype="bf16", gradient_checkpointing=True,
    )
    trainer = LowVramTrainer(pipe, cfg)
    cache = build_cache(args.data, trainer, resolution, verbose=False)
    del trainer.pipe.vae, trainer.pipe.text_encoder, trainer.pipe.text_encoder_2
    torch.cuda.empty_cache()

    trainer.setup()
    stats = trainer.train(cache[: args.images], log_every=1000)
    vram = peak_vram_gb()
    del trainer, pipe
    torch.cuda.empty_cache()
    return vram, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--resolutions", default="512,768")
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    rows = []
    for resolution in [int(r) for r in args.resolutions.split(",")]:
        print("== %d px ==" % resolution, flush=True)
        try:
            vram, stats = run(args, resolution)
            per_it = stats["seconds"] / max(stats["steps"], 1)
            rows.append({"resolution": resolution, "peak_vram_gb": round(vram, 2),
                         "s_per_it": round(per_it, 2), "steps": stats["steps"],
                         "status": "ok"})
            print("   peak_vram=%.2f GB  %.2f s/it" % (vram, per_it), flush=True)
        except RuntimeError as exc:
            rows.append({"resolution": resolution, "status": "oom", "error": str(exc)[:120]})
            print("   failed: %s" % str(exc)[:120], flush=True)

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
