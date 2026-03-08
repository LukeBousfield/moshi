# Moshi: Command-Line Arguments Reference

This document covers all CLI arguments for the moshi scripts, with custom/added arguments highlighted.

---

## `moshi/moshi/server.py` — WebSocket Server (Real-Time Chat)

Run with: `python -m moshi.server [OPTIONS]`

### Server

| Argument | Type | Default | Description |
|---|---|---|---|
| `--host` | str | `localhost` | Server host |
| `--port` | int | `8998` | Server port |
| `--static` | str | None | Path to static content directory for the web client |
| `--ssl` | str | None | Directory containing `key.pem` and `cert.pem` for HTTPS |
| `--gradio-tunnel` | flag | off | Activate a gradio tunnel for remote access |
| `--gradio-tunnel-token` | str | None | Custom secret token to keep a stable tunnel URL |

### Model Loading

| Argument | Type | Default | Description |
|---|---|---|---|
| `--hf-repo` | str | `DEFAULT_REPO` (Moshiko) | HuggingFace repo for pretrained model weights |
| `--tokenizer` | str | None | Path to a local tokenizer file |
| `--moshi-weight` | str | None | Path to a local Moshi checkpoint |
| `--mimi-weight` | str | None | Path to a local Mimi checkpoint |
| `--config-path` | str | None | Path to a local config JSON file |
| `--device` | str | `cuda` | Device to run on (`cuda`, `cpu`) |
| `--half` | flag | bfloat16 | Use float16 instead of bfloat16 (better for older GPUs) |
| `--cfg-coef` | float | `1.0` | Classifier-free guidance coefficient |

### Custom/Added Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| **`--lora-weight`** | str | None | Path to a LoRA checkpoint file. Loads and applies LoRA adapter weights on top of the base model. |
| **`--no_fuse_lora`** | flag | fuse by default | Do **not** fuse LoRA layers into Linear layers. When omitted, LoRA weights are fused for faster inference. |
| **`--separate-semantic-proj`** | flag | off | Enable separated semantic/acoustic projection layers. Required when serving a model fine-tuned with separated projections. |
| **`--temporal-acoustic-codebooks`** | int | None (all) | Number of acoustic codebooks per stream to feed the Temporal Transformer. `0` = semantic only, `None` = all codebooks. Reduces noise in the temporal representation. |

---

## `moshi/moshi/run_inference.py` — Speech-to-Text Inference

Run with: `python -m moshi.run_inference [OPTIONS] infile [outfile]`

| Argument | Type | Default | Description |
|---|---|---|---|
| `infile` | str | **required** | Input audio file |
| `outfile` | str | `""` (derived from infile) | Output audio file in WAV format |
| `--hf-repo` | str | `DEFAULT_REPO` | HuggingFace repo for pretrained model |
| `--tokenizer` | str | None | Path to a local tokenizer file |
| `--moshi-weight` | str | None | Path to a local Moshi checkpoint |
| `--mimi-weight` | str | None | Path to a local Mimi checkpoint |
| `--config` / `--lm-config` | str | None | Model config as a JSON file |
| `--batch-size` | int | `8` | Batch size for inference |
| `--device` | str | `cuda` | Device to run on |
| `--half` | flag | bfloat16 | Use float16 instead of bfloat16 |
| `--cfg-coef` | float | `1.0` | Classifier-free guidance coefficient |

---

## `moshi/moshi/run_tts.py` — Text-to-Speech Generation

Run with: `python -m moshi.run_tts [OPTIONS] jsonl`

| Argument | Type | Default | Description |
|---|---|---|---|
| `jsonl` | Path | **required** | JSONL file containing TTS requests |
| `--out-folder` | Path | `tts-outputs` | Output folder for generated audio |
| `--hf-repo` | str | `DEFAULT_DSM_TTS_REPO` | HuggingFace repo for TTS model |
| `--voice-repo` | str | `DEFAULT_DSM_TTS_VOICE_REPO` | HuggingFace repo for pre-computed voice embeddings |
| `--config` / `--lm-config` | str | None | Model config as a JSON file |
| `--tokenizer` | str | None | Path to a local tokenizer file |
| `--moshi-weight` | str | None | Path to a local Moshi checkpoint |
| `--mimi-weight` | str | None | Path to a local Mimi checkpoint |
| `--batch-size` | int | `32` | Batch size for inference |
| `--device` | str | `cuda` | Device to run on |
| `--half` | flag | bfloat16 | Use float16 instead of bfloat16 |
| `--cfg-coef` | float | `2.0` | Classifier-free guidance coefficient |
| `--nq` | int | `32` | Number of codebooks to generate |
| `--temp` | float | `0.6` | Sampling temperature for text and audio |
| `--only-wav` | flag | off | Only save audio output (skip tokens/debug info) |

### Padding / Pacing Controls

| Argument | Type | Default | Description |
|---|---|---|---|
| `--max-padding` | int | `8` | Maximum consecutive padding steps |
| `--initial-padding` | int | `2` | Padding steps at the beginning |
| `--final-padding` | int | `4` | Padding steps after the last word |
| `--padding-bonus` | float | `0.0` | Bonus added to padding logits (-2 to 2). Higher = slower speech. |
| `--padding-between` | int | `1` | Minimum fixed padding steps between words |

---

## `scripts/finetune_separated_proj.py` — Fine-Tuning (Custom Script)

Run with: `python scripts/finetune_separated_proj.py [OPTIONS]`

This is an entirely custom script for fine-tuning Moshi with separated semantic/acoustic projection layers.

### Model Source

| Argument | Type | Default | Description |
|---|---|---|---|
| `--hf-repo` | str | None | HuggingFace repo for pretrained model |
| `--checkpoint` | str | None | Path to a local checkpoint file |

### Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | str | **required** | Directory containing tokenized `.safetensors` files |
| `--seq-len` | int | `2048` | Sequence length per training sample |

### Training Hyperparameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--steps` | int | `10000` | Total training steps |
| `--batch-size` | int | `4` | Batch size |
| `--grad-accum` | int | `1` | Gradient accumulation steps |
| `--lr` | float | `1e-4` | Learning rate for projection layers |
| `--attn-lr` | float | `1e-6` | Learning rate for unfrozen attention layers |
| `--weight-decay` | float | `0.01` | Weight decay |
| `--warmup-steps` | int | `500` | Learning rate warmup steps |
| `--max-grad-norm` | float | `1.0` | Maximum gradient norm for clipping (0 to disable) |
| `--semantic-weight` | float | `100.0` | Loss weight for semantic codebooks relative to acoustic codebooks |

### Selective Layer Unfreezing

| Argument | Type | Default | Description |
|---|---|---|---|
| `--unfreeze-attn-layers` | str | None | Comma-separated layer indices to unfreeze for attention fine-tuning (e.g. `28,29,30,31`) |

### Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output-dir` | str | **required** | Output directory for checkpoints |
| `--save-every` | int | `1000` | Save a checkpoint every N steps |
| `--log-every` | int | `50` | Log metrics every N steps |
| `--save-full-model` | flag | off | Save the full model weights, not just the trained projection layers |

### System

| Argument | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `cuda` | Device (`cuda` or `cpu`) |
| `--num-workers` | int | `4` | Number of DataLoader worker processes |
