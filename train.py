#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line entry point for low-VRAM SDXL LoRA training.

Example:
    python train.py --data ./dataset/karisa --output ./out/karisa.safetensors
"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusionXLPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_cache
from src.trainer import LowVramTrainer, peak_vram_gb, reset_peak_vram

DEFAULT_TARGETS = "to_q,to_k,to_v,to_out.0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an SDXL LoRA adapter on a low-VRAM GPU."
    )
    parser.add_argument("--base-model", required=True,
                        help="path to an SDXL .safetensors checkpoint")
    parser.add_argument("--data", required=True,
                        help="folder with images and .txt captions")
    parser.add_argument("--output", required=True, help="output .safetensors")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None,
                        help="defaults to --rank")
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                        help="bf16 is strongly recommended; fp16 often NaNs")
    parser.add_argument("--target-modules", default=DEFAULT_TARGETS)
    parser.add_argument("--trigger-word", default=None,
                        help="prepended to captions without a .txt file")
    parser.add_argument("--fallback-caption", default="1girl")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)
    if args.alpha is None:
        args.alpha = args.rank
    args.target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    return args


def main(argv=None):
    cfg = parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("loading %s (CPU, this takes a while)" % os.path.basename(cfg.base_model))
    pipe = StableDiffusionXLPipeline.from_single_file(
        cfg.base_model, torch_dtype=torch.bfloat16, use_safetensors=True
    )

    trainer = LowVramTrainer(pipe, cfg)
    reset_peak_vram()

    print("pre-computing latents and text embeddings ...")
    cache = build_cache(cfg.data, trainer, cfg.resolution,
                        fallback_caption=cfg.fallback_caption,
                        trigger_word=cfg.trigger_word)

    del trainer.pipe.vae, trainer.pipe.text_encoder, trainer.pipe.text_encoder_2
    torch.cuda.empty_cache()

    trainable = trainer.setup()
    print("trainable parameters: %.2f M" % (trainable / 1e6))

    stats = trainer.train(cache, log_every=cfg.log_every)
    fmt, count = trainer.save(cfg.output)

    print("")
    print("done  steps=%d  time=%.0fs  %.2f s/it  peak_vram=%.2f GB"
          % (stats["steps"], stats["seconds"],
             stats["seconds"] / max(stats["steps"], 1), peak_vram_gb()))
    print("saved (%s format, %d tensors) -> %s" % (fmt, count, cfg.output))


if __name__ == "__main__":
    main()
