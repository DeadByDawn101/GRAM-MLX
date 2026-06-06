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

- GRAM paper: Baek, Jo, Kim, Ren, Bengio, Ahn (KAIST/NYU/Mila)
- MLX port: [@DeadByDawn101](https://github.com/DeadByDawn101) / RavenX LLC
- OpenMythos-MLX: [github.com/DeadByDawn101/OpenMythos-MLX](https://github.com/DeadByDawn101/OpenMythos-MLX)

## License

MIT

> *"We don't give up. We do what others don't and build what isn't possible."* — RavenX LLC
