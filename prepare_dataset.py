#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalise a folder of images into the training layout.

Creates <out>/<name>.jpg and <out>/<name>.txt (caption) for every image,
optionally writing a shared trigger word into each caption.
"""

import argparse
import os

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source folder")
    parser.add_argument("--dst", required=True, help="output folder")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--trigger-word", default=None)
    parser.add_argument("--caption", default=None,
                        help="caption used when a .txt is missing")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(args.src)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        src = os.path.join(args.src, name)
        stem = os.path.splitext(name)[0]
        image = Image.open(src).convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((args.size, args.size), Image.LANCZOS)
        image.save(os.path.join(args.dst, stem + ".jpg"), quality=95)

        caption_path = os.path.join(args.src, stem + ".txt")
        if os.path.exists(caption_path):
            with open(caption_path, encoding="utf-8") as handle:
                caption = handle.read().strip()
        else:
            caption = args.caption or ""
            if args.trigger_word:
                caption = (args.trigger_word + ", " + caption).strip(", ")
        if caption:
            with open(os.path.join(args.dst, stem + ".txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(caption)
        count += 1
        print("  %s -> %s.jpg" % (name, stem))
    print("done, %d images" % count)


if __name__ == "__main__":
    main()
