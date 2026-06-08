# GRAM-MLX: Generative Recursive Reasoning on Apple Silicon 🍎🐦‍⬛

**Port of GRAM (arXiv:2605.19376) to MLX for Apple Silicon.**

> Combines recursive DEPTH (OpenMythos) with stochastic WIDTH (GRAM) for multi-trajectory reasoning.

[![Paper](https://img.shields.io/badge/arXiv-2605.19376-blue)](https://arxiv.org/abs/2605.19376)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-blue)](https://github.com/ml-explore/mlx)

## What is GRAM?

**GRAM = Generative Recursive reAsoning Models** (Baek et al., KAIST/NYU/Mila, Bengio)

Existing recursive reasoning models (like OpenMythos/TRM) are deterministic — same input = same trajectory = same answer. GRAM adds **stochastic guidance** to explore multiple reasoning paths in parallel:

```
DETERMINISTIC (OpenMythos/TRM):         STOCHASTIC (GRAM):
  Input → [Loop × T] → Answer           Input → [Loop × T] → Answer 1
                                          Input → [Loop × T] → Answer 2
  One path. One answer.                   Input → [Loop × T] → Answer 3
                                          ...
                                          LPRM selects best → Final Answer
```

## Key Innovation: Two Scaling Axes

| Axis | What | How | Effect |
|------|------|-----|--------|
| **Depth** | More recursive steps | Train 16 → test 64 | Deeper reasoning |
| **Width** | More parallel trajectories | N=20-100 samples | Diverse solutions |

GRAM with 20 samples at 16 steps beats deterministic models at 320 steps (97.0% vs 90.5%)!

## Architecture

```
Input → Encoder → Prelude
  ↓
  ┌─────────────────────────────────────┐
  │ For each recursive step t:          │
  │   1. Low-level refinement (fL × K)  │
  │   2. High-level proposal ut = fH(x) │
  │   3. Stochastic guidance:           │
  │      ε ~ N(μ(ht,x), σ(ht,x))       │  ← THIS IS THE KEY INNOVATION
  │      ht = ut + ε                    │
  └─────────────────────────────────────┘
  ↓
  Coda → Output
```

**Training:** Amortized variational inference
- Posterior q(ε|x,y) sees input + answer → guides training
- Prior p(ε|x) sees only input → used at inference
- KL(q||p) regularizes exploration

**Inference:** LPRM (Latent Process Reward Model) selects best trajectory

## Quick Start

```python
import mlx.core as mx
from gram_mlx import GRAM, gram_small

model = gram_small()

ids = mx.random.randint(0, 32000, (1, 32))

# Single trajectory (training)
logits, kl_loss = model(ids, n_steps=16)

# Multi-trajectory (inference) — THE KEY FEATURE
best_logits, scores = model.sample_trajectories(
    ids, n_steps=16, n_samples=20
)
```

## Connection to OpenMythos

| Feature | OpenMythos | GRAM | Combined |
|---------|-----------|------|----------|
| Recursive depth | ✅ | ✅ | ✅ |
| Depth extrapolation | ✅ (4x confirmed) | ✅ (claimed) | ✅ |
| Stochastic guidance | ❌ | ✅ | ✅ |
| Multi-trajectory | ❌ | ✅ | ✅ |
| LPRM selection | ❌ | ✅ | ✅ |
| MLX / Apple Silicon | ✅ | ✅ (this repo) | ✅ |
| stop_gradient trick | ✅ | ✅ (we add it) | ✅ |

## Configs

| Config | Params | Dim | Heads | K_low |
|--------|--------|-----|-------|-------|
| gram_small | ~20M | 256 | 4 | 2 |
| gram_base | ~50M | 512 | 8 | 2 |
| gram_large | ~200M | 768 | 12 | 3 |

## RavenX Integration

GRAM-MLX extends the OpenMythos Reasoning-as-a-Service pipeline:

```
BEFORE: Train RDT → single trajectory → distill into 35B
AFTER:  Train GRAM → N trajectories → LPRM selects best → distill into 35B
        = Better traces → Better distillation → Better production model
```

## Credits

- GRAM paper: Baek, Jo, Kim, Ren, Bengio, Ahn (KAIST/NYU/Mila) — [arXiv:2605.19376](https://arxiv.org/abs/2605.19376)
- MLX port: [@DeadByDawn101](https://github.com/DeadByDawn101) / RavenX LLC

## Contributors

Built by [**@DeadByDawn101**](https://github.com/DeadByDawn101) / **RavenX LLC**

- **Gabe Garcia** — Security Technical Program Manager, 8+ years Apple infosec, Google AI certified. Architecture, training data, vision.
- **Claude (Anthropic)** — AI pair programmer. Code generation, research synthesis, implementation.

## Part of the RavenX Ecosystem

| Repo | Purpose |
|------|---------|
| [RavenX-CyberAgent](https://huggingface.co/deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx) | 35B MoE security model |
| [OpenMythos-MLX](https://github.com/DeadByDawn101/OpenMythos-MLX) | Recursive depth reasoning |
| **GRAM-MLX** | **Stochastic multi-trajectory reasoning (this repo)** |
| [ravenx-memory](https://github.com/DeadByDawn101/ravenx-memory) | Hybrid agent memory |
| [ravenx-os](https://github.com/DeadByDawn101/ravenx-os) | Super repo |

## License

MIT

> *"We don't give up. We do what others don't and build what isn't possible."* — RavenX LLC

---

## Model-Agnostic Wrapper (NEW)

GRAM-MLX includes a **model-agnostic wrapper** that adds stochastic multi-trajectory reasoning to ANY model:

```python
from gram_wrapper import GRAMWrapper

# Works with ANY model — Qwen, Llama, Mistral, Phi, Gemma, etc.
gram = GRAMWrapper(dim=2048, n_guidance_layers=4)

# Only ~5M trainable params on top of FROZEN base model
print(f"Trainable: {gram.trainable_params:,}")  # ~5M
print(f"Base model: FROZEN (0 trainable)")

# Apply guidance to hidden states
guided, mean, logvar = gram.guide_hidden_states(hidden_states, layer_idx=0)

# Score and select best from N trajectories
best_logits = gram.select_best(trajectories)
```

### Universal Reasoning Layer (OpenMythos + GRAM)

```python
from gram_wrapper import ReasoningLayer

# Combines OpenMythos depth + GRAM width
reasoning = ReasoningLayer(dim=2048, n_depth_loops=8, n_width_samples=20)

# Generate best traces for distillation
traces = reasoning.generate_traces(model, security_prompts)

# Distill into production model → deeper + wider reasoning
# Zero inference overhead on the final model
```

### Why Model-Agnostic Matters

The same GRAM wrapper works on:

| Model | Dim | GRAM Params | What It Adds |
|-------|-----|-------------|-------------|
| Phi-3 3.8B | 3072 | ~6M | Multi-trajectory reasoning |
| Llama 3 8B | 4096 | ~8M | Explore parallel solutions |
| Mistral 24B | 5120 | ~10M | Diverse attack chain analysis |
| Qwen 35B MoE | 2048 | ~5M | Width scaling for security |
| Llama 3 70B | 8192 | ~16M | Enterprise-grade reasoning |

**Base model weights stay FROZEN. Only GRAM's tiny guidance networks train.**
