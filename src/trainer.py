# -*- coding: utf-8 -*-
"""Low-VRAM SDXL LoRA trainer core.

Key idea: instead of loading the whole pipeline on the GPU, every frozen
component (VAE / text encoders / UNet) is moved to the device only while it is
needed and put back to CPU afterwards. Combined with cached latents and text
embeddings this keeps peak VRAM far below what a naive implementation needs.
"""

import math
import time

import torch
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model


def build_lora_config(rank, alpha, target_modules):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
    )


class LowVramTrainer:
    """Train a LoRA adapter on a frozen SDXL UNet."""

    def __init__(self, pipeline, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

        self.unet = pipeline.unet
        self.pipe = pipeline

        for module in (pipeline.unet, pipeline.vae, pipeline.text_encoder,
                       pipeline.text_encoder_2):
            module.requires_grad_(False)
            module.eval()

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def to_cpu(self, module):
        module.to("cpu")
        torch.cuda.empty_cache()

    def to_device(self, module):
        module.to(self.device, self.dtype)
        return module

    @torch.no_grad()
    def encode_prompt(self, caption):
        """Encode a caption, shuttling each text encoder to the GPU alone."""
        outs = []
        for tokenizer, encoder in ((self.pipe.tokenizer, self.pipe.text_encoder),
                                   (self.pipe.tokenizer_2, self.pipe.text_encoder_2)):
            self.to_device(encoder)
            ids = tokenizer(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            out = encoder(ids, output_hidden_states=True)
            outs.append(out.hidden_states[-2].cpu())
            self.to_cpu(encoder)
        emb = torch.cat(outs, dim=-1)
        pooled = outs[1][:, -1, :]
        return emb, pooled

    @torch.no_grad()
    def encode_image(self, image_tensor):
        """Encode an image into latents with the VAE."""
        self.to_device(self.pipe.vae)
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor
        self.to_cpu(self.pipe.vae)
        return latents.cpu()

    def setup(self):
        cfg = self.cfg
        lora_cfg = build_lora_config(cfg.rank, cfg.alpha, cfg.target_modules)
        self.unet = get_peft_model(self.unet, lora_cfg)
        if cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        self.to_device(self.unet)
        self.unet.train()

        self.params = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        return sum(p.numel() for p in self.params)

    def train(self, cache, log_every=5):
        """Run the training loop over a cached dataset."""
        cfg = self.cfg
        total = len(cache) * cfg.steps_per_image * cfg.epochs
        step = 0
        started = time.time()
        losses = []

        for epoch in range(cfg.epochs):
            torch.manual_seed(cfg.seed + epoch)
            order = torch.randperm(len(cache)).tolist()
            for index in order:
                item = cache[index]
                for _ in range(cfg.steps_per_image):
                    latents = item["latents"].to(self.device, self.dtype)
                    embeds = item["embeds"].to(self.device, self.dtype)
                    pooled = item["pooled"].to(self.device, self.dtype)

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 1000, (1,), device=self.device
                    ).long()
                    noisy = self.scheduler.add_noise(latents, noise, timesteps)

                    added_cond = {
                        "text_embeds": pooled,
                        "time_ids": torch.tensor(
                            [[cfg.resolution, cfg.resolution, 0, 0,
                              cfg.resolution, cfg.resolution]],
                            device=self.device, dtype=self.dtype,
                        ),
                    }
                    pred = self.unet(
                        noisy, timesteps,
                        encoder_hidden_states=embeds,
                        added_cond_kwargs=added_cond,
                    ).sample
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

                    self.optimizer.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        step += 1
                        losses.append(float("nan"))
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                    self.optimizer.step()

                    step += 1
                    losses.append(loss.item())
                    if step % log_every == 0 or step == total:
                        elapsed = time.time() - started
                        print(
                            "step %d/%d  loss=%.4f  %.0fs  %.2fs/it"
                            % (step, total, loss.item(), elapsed, elapsed / step),
                            flush=True,
                        )
            torch.cuda.empty_cache()

        return {
            "steps": step,
            "seconds": time.time() - started,
            "losses": losses,
        }

    def save(self, path):
        """Save the adapter. Try kohya format first, fall back to diffusers."""
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        state = get_peft_model_state_dict(self.unet)
        fmt = "diffusers"
        try:
            from diffusers.utils.state_dict_utils import (
                convert_state_dict_to_diffusers,
                convert_state_dict_to_kohya,
            )
            state = convert_state_dict_to_diffusers(state)
            state = convert_state_dict_to_kohya(state)
            fmt = "kohya"
        except Exception as exc:  # kohya conversion is optional
            print("kohya conversion unavailable, saving diffusers format:", exc)

        state = {k: v.to("cpu").contiguous() for k, v in state.items()}
        save_file(
            state,
            path,
            metadata={
                "ss_network_dim": str(self.cfg.rank),
                "ss_network_alpha": str(self.cfg.alpha),
            },
        )
        return fmt, len(state)


def peak_vram_gb(device="cuda"):
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def reset_peak_vram(device="cuda"):
    torch.cuda.reset_peak_memory_stats(device)


def count_trainable(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
