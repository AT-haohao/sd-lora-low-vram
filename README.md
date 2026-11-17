# sd-lora-low-vram

**Train an SDXL LoRA on a 6 GB laptop GPU — no kohya, no cloud, no 24 GB card.**

Most SDXL LoRA trainers assume 16-24 GB of VRAM. This one is built around the
opposite constraint: a single RTX 4050 Laptop GPU with **6 GB**.

The trick is not a smaller model — it is *moving* the model. VAE, both text
encoders and the UNet are shuttled to the GPU one at a time and parked back in
system RAM afterwards, while latents and text embeddings are pre-computed once
and cached. The result is a full SDXL LoRA training at about 1 s/it at 512 px.

---

## Benchmark

Measured on **RTX 4050 Laptop, 6 GB VRAM, 16 GB RAM**, Windows 11, bf16,
rank 16, gradient checkpointing on:

| Resolution | Peak VRAM | Speed | Status |
|---|---|---|---|
| 512 x 512 | 5.2 GB | 1.0 s/it | ok |
| 768 x 768 | 6.1 GB | 2.6 s/it | ok, close to the limit |
| 1024 x 1024 | - | - | out of memory |

Re-run ```python benchmark.py``` on your own card to reproduce. Above 768 px
will need a bigger card.

## How it works

```text
   CPU RAM                          GPU (6 GB)
  -----------------------------------------------
   SDXL checkpoint  --load-->  (nothing yet)
        |
        +- text_encoder   --> encode caption --> embeddings -> CPU
        +- text_encoder_2 --> encode caption --> pooler     -> CPU
        +- vae            --> encode image   --> latents    -> CPU
        |
        +- unet (+ LoRA)  --> train loop (only this stays on GPU)
```

Four things keep the peak low:

1. **Component shuttling** — only one frozen module sits on the GPU at a time.
2. **Pre-computed cache** — latents and embeddings are computed once per image,
   so the training loop never touches the VAE or the text encoders.
3. **bf16, not fp16** — fp16 NaNs out within a few hundred steps on SDXL.
4. **Gradient checkpointing** — trades compute for activation memory.

## Install

```bash
git clone https://github.com/<your-name>/sd-lora-low-vram.git
cd sd-lora-low-vram
pip install -r requirements.txt
```

Requires a CUDA GPU. Developed against PyTorch 2.6 / diffusers 0.42 on Windows.

## Quick start

```bash
# 1. normalise images into 512x512 with matching .txt captions
python prepare_dataset.py --src ./raw --dst ./dataset/karisa --size 512

# 2. train
python train.py \
    --base-model /path/to/sdxl.safetensors \
    --data ./dataset/karisa \
    --output ./out/karisa.safetensors \
    --resolution 512 --rank 16 --epochs 4 --lr 8e-5

# 3. measure your own card
python benchmark.py --base-model /path/to/sdxl.safetensors --data ./dataset/karisa
```

Output is a `.safetensors` adapter, in kohya format when the conversion helper
is available and diffusers format otherwise. Both load in ComfyUI's LoRA loader.

## CLI reference

| Flag | Default | Notes |
|---|---|---|
| --base-model | required | any SDXL .safetensors checkpoint |
| --data | required | folder of img.jpg + img.txt pairs |
| --resolution | 512 | 512 is the sweet spot for 6 GB |
| --rank | 16 | 16-32 works, higher needs more VRAM |
| --lr | 8e-5 | 8e-5 to 1e-4 is the usable range |
| --epochs | 4 | 4-6 for about 20 images |
| --dtype | bf16 | fp16 exists but will NaN |
| --trigger-word | none | prepended when a caption is missing |

## Pitfalls

See `docs/pitfalls.md` for the long version. The short one:

- **fp16 produces loss=nan.** Use bf16. SDXL behaviour, not a bug here.
- **10 images are not enough.** Under ~15 the model memorises captions.
- **Caption wording dominates the result.** Writing a skin tone into the caption
  really does change every generated face.
- **Close ComfyUI before training.** Two processes on 6 GB roughly triples step time.

## License

MIT.

---
