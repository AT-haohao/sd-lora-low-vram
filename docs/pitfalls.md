# Pitfalls

Collected while training character LoRAs on a 6 GB RTX 4050 Laptop.

## 1. fp16 NaNs out

With `torch.float16` the loss goes to `nan` after a few hundred steps and never
recovers. bf16 has enough exponent range for SDXL. The trainer skips non-finite
losses as a safety net, but skipping is not a fix — switch dtype.

## 2. Captions matter more than hyperparameters

Three caption experiments, each verified against the output images:

| Caption said | Result |
|---|---|
| dark skin, tan skin | every generated face came out several shades too dark |
| light skin, mature female, young woman | correct skin tone and age |
| amber eyes | solid orange irises, no texture |
| grey eyes, gold flecks in eyes, grey iris with golden sparks | grey iris with visible specks |

Rule: describe the target state, not the observed state. Two or three
descriptive clauses per feature beat one vague word.

## 3. Too few images

Under about 15 images the adapter overfits the captions instead of the subject.
At 38 images and 5 epochs the identity held across poses and outfits.

## 4. Running ComfyUI at the same time

Both processes want the GPU. Training dropped from 1.0 s/it to 2.6 s/it when
ComfyUI was left running. Close it first.

## 5. Style and identity bleed together

A LoRA trained on 8 images of one character in one art style carries both. To
reuse the style for a different character you need images of a different
character in that style.

## 6. kohya conversion is optional

diffusers can fail to convert with "Original type None is not supported". The
trainer falls back to diffusers format and ComfyUI still loads it.
