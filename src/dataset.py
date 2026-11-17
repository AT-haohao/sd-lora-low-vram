# -*- coding: utf-8 -*-
"""Dataset helpers: load image/caption pairs and pre-compute latents."""

import os

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def list_pairs(data_dir):
    """Return [(image_path, caption_path)] for every image in data_dir."""
    pairs = []
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        image_path = os.path.join(data_dir, name)
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        pairs.append((image_path, caption_path))
    return pairs


def read_caption(caption_path, fallback, trigger_word=None):
    if os.path.exists(caption_path):
        with open(caption_path, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text:
            return text
    if trigger_word:
        return trigger_word + ", " + fallback
    return fallback


def load_image(path, resolution):
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS
    )
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def build_cache(data_dir, trainer, resolution, fallback_caption="1girl",
                trigger_word=None, verbose=True):
    """Pre-compute text embeddings and latents once, then free the encoders."""
    pairs = list_pairs(data_dir)
    if not pairs:
        raise RuntimeError("no images found in %s" % data_dir)

    cache = []
    for index, (image_path, caption_path) in enumerate(pairs, 1):
        caption = read_caption(caption_path, fallback_caption, trigger_word)
        embeds, pooled = trainer.encode_prompt(caption)
        latents = trainer.encode_image(
            load_image(image_path, resolution).to(trainer.device, trainer.dtype)
        )
        cache.append({"latents": latents, "embeds": embeds, "pooled": pooled})
        if verbose:
            print("  cached %d/%d  %s" % (index, len(pairs),
                                          os.path.basename(image_path)),
                  flush=True)
    return cache
