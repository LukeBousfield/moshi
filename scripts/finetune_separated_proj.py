#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# Fine-tuning script for separated semantic/acoustic projections in Moshi.
#
# This script loads a pretrained Moshi checkpoint, enables the separated
# semantic/acoustic projection layers, and fine-tunes them (+ optionally
# some attention layers) while keeping the rest of the model frozen.
#
# Usage:
#   python -m scripts.finetune_separated_proj \
#       --hf-repo kyutai/moshiko-pytorch-bf16 \
#       --data-dir /path/to/tokenized/data \
#       --output-dir /path/to/output \
#       --steps 10000 \
#       --lr 1e-4

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader, Dataset

from moshi.models.loaders import CheckpointInfo, get_moshi_lm  # noqa: E402
from moshi.utils.utils import cross_entropy  # noqa: E402

logger = logging.getLogger(__name__)


class TokenizedAudioDataset(Dataset):
    """Dataset of pre-tokenized audio codes stored as safetensors files.

    Each file should contain a tensor 'codes' of shape [K, T] where K is the
    number of codebooks (17 for Moshi: 1 text + 16 audio) and T is the number
    of time steps.
    """

    def __init__(self, data_dir: str, seq_len: int = 2048):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.files = sorted(self.data_dir.glob("*.safetensors"))
        if not self.files:
            raise RuntimeError(f"No .safetensors files found in {data_dir}")
        logger.info(f"Found {len(self.files)} data files in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = load_file(str(self.files[idx]))
        codes = data["codes"]  # [K, T]
        K, T = codes.shape
        if T > self.seq_len:
            start = torch.randint(0, T - self.seq_len, (1,)).item()
            codes = codes[:, start:start + self.seq_len]
        elif T < self.seq_len:
            # Pad with zeros (padding token)
            pad = torch.zeros(K, self.seq_len - T, dtype=codes.dtype)
            codes = torch.cat([codes, pad], dim=1)
        return codes


def setup_model(args) -> torch.nn.Module:
    """Load pretrained model with separated projections enabled."""
    if args.hf_repo:
        checkpoint = CheckpointInfo.from_hf_repo(args.hf_repo)
        model = checkpoint.get_moshi(
            device=args.device,
            dtype=torch.bfloat16,
            lm_kwargs_overrides={"separate_semantic_proj": True},
        )
    elif args.checkpoint:
        model = get_moshi_lm(
            args.checkpoint,
            device=args.device,
            dtype=torch.bfloat16,
            lm_kwargs_overrides={"separate_semantic_proj": True},
        )
    else:
        raise ValueError("Must provide either --hf-repo or --checkpoint")
    return model


def freeze_and_configure(model, args):
    """Freeze all parameters, then selectively unfreeze projections and optional attention layers."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the new projection layers
    assert model.semantic_proj is not None, "Model must have separate_semantic_proj=True"
    model.semantic_proj.weight.requires_grad = True
    model.acoustic_proj.weight.requires_grad = True

    trainable_params = [
        {"params": [model.semantic_proj.weight, model.acoustic_proj.weight], "lr": args.lr},
    ]

    # Optionally unfreeze some attention layers at a lower LR
    if args.unfreeze_attn_layers:
        layer_indices = [int(x) for x in args.unfreeze_attn_layers.split(",")]
        attn_params = []
        for idx in layer_indices:
            layer = model.transformer.layers[idx]
            for param in layer.self_attn.parameters():
                param.requires_grad = True
                attn_params.append(param)
        if attn_params:
            trainable_params.append({
                "params": attn_params,
                "lr": args.attn_lr,
            })
        logger.info(f"Unfreezing attention in layers {layer_indices} with lr={args.attn_lr}")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {total_trainable:,} / {total_params:,} params "
                f"({100 * total_trainable / total_params:.2f}%)")

    return trainable_params


def compute_loss(model, codes, semantic_weight: float = 100.0):
    """Run forward pass and compute weighted cross-entropy loss."""
    output = model(codes)

    # Text loss
    text_ce = cross_entropy(output.text_logits, codes[:, :1], output.text_mask)
    text_loss = (text_ce * output.text_mask).sum() / output.text_mask.sum().clamp(min=1)

    # Audio loss — weight semantic codebook (first per stream) higher
    audio_targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q]
    audio_ce = cross_entropy(output.logits, audio_targets, output.mask)

    # Build per-codebook weights: semantic codebooks get higher weight
    B, K, T = audio_ce.shape
    cb_weights = torch.ones(K, device=audio_ce.device)
    for i in range(K):
        if i in model._semantic_codebook_indices:
            cb_weights[i] = semantic_weight
    cb_weights = cb_weights.view(1, K, 1)

    weighted_ce = audio_ce * cb_weights
    audio_loss = (weighted_ce * output.mask).sum() / (output.mask * cb_weights).sum().clamp(min=1)

    loss = text_loss + audio_loss
    return loss, text_loss.item(), audio_loss.item()


def train(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading model...")
    model = setup_model(args)
    model.train()

    trainable_params = freeze_and_configure(model, args)
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Loading dataset...")
    dataset = TokenizedAudioDataset(args.data_dir, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    accum_loss = 0.0
    accum_text_loss = 0.0
    accum_audio_loss = 0.0

    logger.info(f"Starting training for {args.steps} steps...")
    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            codes = batch.to(args.device)  # [B, K, T]
            loss, text_loss, audio_loss = compute_loss(
                model, codes, semantic_weight=args.semantic_weight
            )

            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            accum_text_loss += text_loss / args.grad_accum
            accum_audio_loss += audio_loss / args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"step {step}/{args.steps} | loss={accum_loss:.4f} "
                    f"text={accum_text_loss:.4f} audio={accum_audio_loss:.4f} lr={lr:.2e}"
                )
                accum_loss = 0.0
                accum_text_loss = 0.0
                accum_audio_loss = 0.0

            if step % args.save_every == 0:
                save_checkpoint(model, output_dir, step, args)

    # Final save
    save_checkpoint(model, output_dir, step, args)
    logger.info(f"Training complete. Checkpoints saved to {output_dir}")


def save_checkpoint(model, output_dir: Path, step: int, args):
    """Save projection weights (and optionally full model)."""
    proj_state = {
        "semantic_proj.weight": model.semantic_proj.weight.data,
        "acoustic_proj.weight": model.acoustic_proj.weight.data,
    }
    proj_path = output_dir / f"proj_weights_step{step}.safetensors"
    save_file(proj_state, str(proj_path))
    logger.info(f"Saved projection weights to {proj_path}")

    if args.save_full_model:
        full_path = output_dir / f"model_step{step}.safetensors"
        state_dict = {k: v.data for k, v in model.state_dict().items()}
        save_file(state_dict, str(full_path))
        logger.info(f"Saved full model to {full_path}")

    # Save config for reproducibility
    config_path = output_dir / "finetune_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Moshi with separated semantic/acoustic projections")

    # Model source
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo for pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint")

    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with tokenized .safetensors files")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per sample")

    # Training
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for projection layers")
    parser.add_argument("--attn-lr", type=float, default=1e-6, help="Learning rate for unfrozen attention layers")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--semantic-weight", type=float, default=100.0,
                        help="Loss weight for semantic codebooks vs acoustic")

    # Selective unfreezing
    parser.add_argument("--unfreeze-attn-layers", type=str, default=None,
                        help="Comma-separated layer indices to unfreeze attention (e.g. '28,29,30,31')")

    # Output
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-full-model", action="store_true", help="Save full model (not just projections)")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
