# Moshi: Architecture Summary

> **Paper**: "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037v2)
> **Authors**: Alexandre Defossez*, Laurent Mazare*, Manu Orsini, Amelie Royer, Patrick Perez, Herve Jegou, Edouard Grave, Neil Zeghidour* (Kyutai)
> **License**: CC BY-NC-SA 4.0

## Overview

Moshi is a **full-duplex, real-time speech-to-speech dialogue model**. Unlike traditional spoken dialogue systems that pipeline ASR -> LLM -> TTS, Moshi casts spoken dialogue as direct speech-to-speech generation. It simultaneously listens and speaks via two parallel audio streams, removing the concept of explicit speaker turns.

**Theoretical latency**: 160ms (200ms in practice on an L4 GPU).

---

## High-Level Architecture

Moshi has three core components:

1. **Helium** - A 7B-parameter text language model backbone
2. **Mimi** - A streaming neural audio codec that tokenizes audio
3. **The Moshi model itself** - A multi-stream, hierarchical audio language model combining Helium with a smaller Depth Transformer

```
User Audio ──> [Mimi Encoder] ──> Audio Tokens (user stream)
                                          │
                                          v
                              ┌─────────────────────┐
                              │  Temporal Transformer │  (7B, 32 layers)
                              │     (from Helium)     │
                              └──────────┬────────────┘
                                         │
                                         v
                              ┌─────────────────────┐
                              │   Depth Transformer   │  (small, 6 layers)
                              │   (Depformer / RQ)    │
                              └──────────┬────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    v                    v                    v
              Text Tokens         Semantic Token       Acoustic Tokens
             (Inner Monologue)    (Moshi stream)       (Moshi stream)
                                         │
                                         v
                              [Mimi Decoder] ──> Moshi Audio Output
```

---

## 1. Helium (Text LLM Backbone)

A standard autoregressive Transformer LLM, trained from scratch on English text.

| Parameter | Value |
|---|---|
| Parameters | ~7B |
| Dimension | 4,096 |
| MLP dimension | 11,264 |
| Attention heads | 32 |
| Layers | 32 |
| Context length | 4,096 tokens (text) / 3,000 steps (~4 min for Moshi) |
| Vocab size | 32,000 (SentencePiece unigram) |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Normalization | RMSNorm |
| Activation | SiLU (via Gated Linear Units) |
| Attention | FlashAttention |

Helium serves as the **Temporal Transformer** in Moshi, providing reasoning and knowledge capabilities. It is initialized from the text-pretrained weights when training Moshi.

---

## 2. Mimi (Neural Audio Codec)

Mimi converts raw audio waveforms into discrete tokens and back. It uniquely combines **semantic** and **acoustic** information in a single tokenizer via a split residual vector quantizer (RVQ) with knowledge distillation from WavLM.

### Codec Specs

| Parameter | Value |
|---|---|
| Input | 24 kHz mono audio |
| Frame rate | 12.5 Hz (80ms per frame, 1,920 samples) |
| Bandwidth | ~1.1 kbps |
| Codebooks | 8 used for Moshi (32 total supported) |
| Codebook size | 2,048 entries each |
| Latency | 80ms (single frame, fully streaming/causal) |

### Encoder Architecture (SEANet)

- 1D convolutional encoder with stride ratios [8, 6, 5, 4] (total downsampling: 960x)
- Residual blocks with dilated causal convolutions
- ELU activation, 64 base filters, dimension 512
- **Transformer bottleneck**: 8-layer causal Transformer (d=512, 8 heads, context=250) between encoder and quantizer

### Split RVQ (Key Innovation)

- **First codebook**: Encodes **semantic** information, distilled from WavLM (a self-supervised speech model). This gives the first token strong phonetic discriminability (8.1% ABX error rate), critical for intelligible speech generation.
- **Remaining codebooks (2-8)**: Encode **acoustic** details (speaker identity, prosody, recording conditions).
- The split allows hierarchical generation: predict semantic tokens first, then acoustic tokens.

### Decoder Architecture (SEANet)

- Mirror of encoder with transposed convolutions
- Transformer in decoder as well (8 layers)
- Adversarial-only training (no reconstruction loss on raw waveform)

### Training Details

- Trained on audio data with adversarial losses only (multi-scale STFT discriminator)
- Quantization applied only 50% of the time during training (quantization rate trick) to improve reconstruction quality
- Knowledge distillation from WavLM applied to the semantic (first) codebook

---

## 3. The Moshi Model (Multi-Stream Audio LM)

### Dual-Transformer Hierarchy (RQ-Transformer)

Moshi uses two Transformers in a hierarchical arrangement:

**Temporal Transformer** (large, initialized from Helium):
- 32 layers, dim 4096, 32 heads
- Processes the sequence across **time steps**
- Context: 3,000 steps (~4 minutes of audio)
- Handles temporal dependencies and reasoning

**Depth Transformer** (small, randomly initialized):
- 6 layers, dim 1024, 16 heads, MLP dim 4096
- Processes tokens **within each time step** (across codebooks)
- Models inter-codebook dependencies (text -> semantic -> acoustic)
- No temporal context (only sees current step + Temporal Transformer output)

At each time step, the Temporal Transformer produces a representation. The Depth Transformer then autoregressively predicts all sub-tokens for that step: first the text token, then semantic token, then 7 acoustic tokens for each of the two streams.

### Multi-Stream Design

Moshi models **two parallel audio streams** simultaneously:
- **Stream 1 (Moshi)**: Moshi's own speech output
- **Stream 2 (User)**: The user's incoming audio

This removes explicit turn-taking. Moshi always listens and always generates (either speech or "natural silence"). At inference time, user stream predictions are ignored since real user audio is used instead.

### Inner Monologue (Text Token Prefix)

A key innovation: Moshi predicts **time-aligned text tokens** as a prefix before audio tokens at each step.

The full sequence of sub-tokens per time step (K = 2Q+1 = 17 total):
1. **W_s**: Text token (aligned to Moshi's speech via Inner Monologue)
2. **A_{s,1}**: Semantic audio token for Moshi
3. **A_{s-τ,2}...A_{s-τ,8}**: Delayed acoustic tokens for Moshi (τ=1 step delay)
4. **A'_{s,1}**: Semantic audio token for User
5. **A'_{s-τ,2}...A'_{s-τ,8}**: Delayed acoustic tokens for User

Benefits of Inner Monologue:
- Dramatically improves linguistic quality and factuality of generated speech
- Leverages the text LLM's knowledge while remaining speech-to-speech
- Enables deriving streaming ASR and streaming TTS as special cases
- Text prediction happens per-frame (compatible with real-time streaming)

### Acoustic Delay Pattern

Acoustic tokens (codebooks 2-8) are **delayed by 1 step** (80ms) relative to semantic tokens. This allows the model to first commit to the semantic content before filling in acoustic details, improving both quality and making generation tractable.

---

## Training Data

### Text Data (for Helium)

- **2.1 trillion tokens** of English text
- 12.5% curated sources: Wikipedia (5 dumps: 2017-2022), Wikibooks, Wikisource, Wikinews, StackExchange, pes2o (scientific articles)
- 87.5% filtered CommonCrawl (10 crawls from 2018-2023)

**Text data filtering pipeline**:
- **Deduplication**: MinHash-based fuzzy dedup + exact URL dedup
- **Language identification**: fastText classifier, English only
- **Quality filtering**: Classifier trained on curated vs. random web data

### Audio Data

| Dataset | Size | Purpose |
|---|---|---|
| Unsupervised audio | 7M hours | Audio pre-training (single stream, English speech) |
| Fisher dataset | 2,000 hours | Multi-stream fine-tuning (separate channels per speaker) |
| Supervised multi-stream | 170 hours | TTS model training (not used directly for Moshi) |
| Synthetic instruct data | 20k+ hours | Instruction fine-tuning (generated via multi-stream TTS) |

**Audio preprocessing**:
- All audio resampled to 24 kHz, downmixed to mono
- Transcribed with Whisper large-v3
- Fisher upsampled from 8 kHz to 24 kHz using AudioSR
- Timestamps obtained with `whisper-timestamped`

### Synthetic Instruct Data

Generated by fine-tuning Helium on Open Hermes + real conversation transcripts, then synthesizing with multi-stream TTS:
- General knowledge conversations (seeded from Wikipedia/StackExchange)
- Voice instruction conversations (speaking styles, emotions)
- Misspelling robustness conversations
- False fact correction conversations
- Math, grammar, trivia Q&A
- Safety conversations (refusal training)
- Moshi's voice: conditioned on a single actor with 70+ speaking styles

---

## Training Pipeline

Training proceeds in 5 stages, all using AdamW optimizer on H100 GPUs with FSDP:

### Stage 1: Helium Pre-training (Text Only)
- 500k steps, batch size 4.2M tokens
- LR: 3e-4 with cosine schedule + linear warmup
- Weight decay: 0.1

### Stage 2: Moshi Pre-training (Single-Stream Audio)
- Initialize Temporal Transformer from Helium; randomly initialize Depth Transformer
- 1M steps, batch size 16h of audio (5 min sequences)
- Single audio stream (all speakers mixed)
- Text tokens masked 30% of the time; text-audio delay randomized [-0.6s, +0.6s]
- 50% of batches are text-only (prevents catastrophic forgetting)
- LR: 3e-5 (Temporal) / 2e-4 (Depth), cosine schedule
- Two separate optimizer states for text vs audio batches

### Stage 3: Moshi Post-training (Simulated Multi-Stream)
- 100k steps, batch size 8h of audio
- Uses PyAnnote diarization to simulate two streams from unsupervised data
- Text-audio delay fixed to 0
- LR: 3e-6 (Temporal) / 5e-5 (Depth), fixed
- 10% text-only batches

### Stage 4: Moshi Fine-tuning (Real Multi-Stream)

**Phase A - Fisher** (real conversations):
- 10k steps, batch size 40 min of audio
- LR: 2e-6 (Temporal) / 4e-6 (Depth)
- Learns real multi-stream dynamics (overlaps, interruptions)

**Phase B - Instruct fine-tuning** (synthetic data):
- 30k steps, batch size 2.7h of audio
- LR: 2e-6 for both transformers
- Data augmentation on user stream:
  - Random gain: [-24dB, +15dB] (50% of the time)
  - Noise addition from DNS challenge (30% of the time, SNR: -30dB to +6dB)
  - Echo simulation: scaled copy of Moshi's stream added to user (factor 0-0.2, delay 100-500ms)
  - Reverb augmentation (30% of the time with echo)

### Stage 5: TTS Training
- Shares audio pre-training with Moshi
- Post-training with 2-second audio delay relative to text
- Fine-tuned on the supervised multi-stream dataset
- Used to generate synthetic instruct data (not used to train Moshi directly)

### Loss Function

Weighted cross-entropy across all sub-tokens:
- Text token and combined audio tokens weighted equally
- Within audio tokens: semantic tokens weighted 100x, acoustic tokens weighted 1x
- Padding tokens down-weighted by 50%

---

## Inference Pipeline

### Real-Time Streaming Loop

```
1. Receive 80ms of user audio (1,920 samples at 24kHz)
2. Encode with Mimi → 1 frame of 8 audio codes (user stream)
3. Feed user codes + previous Moshi codes into Temporal Transformer
4. Depth Transformer predicts: text token → semantic token → 7 acoustic tokens (for Moshi)
5. Decode Moshi's 8 audio codes with Mimi → 80ms of output audio
6. Send output audio to user
```

### Sampling Parameters
- Audio temperature: 0.8
- Text temperature: 0.7
- Top-k sampling for token selection
- Classifier-free guidance (optional)

### Latency Breakdown
- Mimi frame latency: 80ms
- Acoustic delay (1 step): 80ms
- **Total theoretical**: 160ms
- **Practical**: ~200ms (on L4 GPU)

---

## Model Variants & Deployment

### Available Voices
- **Moshika** (female voice)
- **Moshiko** (male voice)

### Quantization Options
| Format | Size | Platform |
|---|---|---|
| bf16 | Full precision | PyTorch, MLX, Rust |
| q8 | 8-bit quantized | PyTorch, MLX, Rust |
| q4 | 4-bit quantized | MLX only |

### Implementation Backends
- **PyTorch** (`moshi/`): Research/development, CUDA graph optimization
- **MLX** (`moshi_mlx/`): Apple Silicon (M1/M3), uses `rustymimi` for codec
- **Rust** (`rust/`): Production deployment, WebSocket server, CUDA/Metal support

### Resource Requirements
- PyTorch (bf16): ~24GB+ GPU VRAM
- Rust/MLX (q8): ~8-12GB VRAM
- MLX (q4): ~2-4GB on M-series Mac

---

## Key Innovations Summary

1. **Full-duplex dialogue**: Two parallel audio streams remove explicit turn-taking
2. **Inner Monologue**: Time-aligned text token prediction as a prefix to audio tokens dramatically improves linguistic quality while maintaining streaming compatibility
3. **Mimi codec**: Single tokenizer combining semantic + acoustic information via split RVQ with WavLM distillation, operating at 12.5Hz / 1.1kbps
4. **RQ-Transformer**: Hierarchical two-transformer design (Temporal + Depth) enables efficient streaming generation of multi-codebook audio
5. **Streaming ASR/TTS for free**: By adjusting the text-audio delay, the same model architecture derives streaming ASR and TTS
6. **Data augmentation for robustness**: Noise, echo, reverb, and gain augmentation during instruct fine-tuning makes the model robust to real-world audio conditions

---

## Safety Measures

- **Toxicity filtering**: Analysis and mitigation of toxic outputs
- **Regurgitation prevention**: Fine-tuning reduces verbatim reproduction of training data
- **Voice consistency**: Single-actor TTS conditioning ensures Moshi doesn't impersonate other voices
- **Watermarking**: Signal-based audio watermarking for identifying AI-generated content
- **Safety conversations**: Training data includes refusal examples for unethical/NSFW requests
