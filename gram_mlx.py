"""
GRAM-MLX: Generative Recursive reAsoning Models for Apple Silicon

Port of GRAM (arXiv:2605.19376) to MLX.
Adds stochastic multi-trajectory reasoning to recursive depth transformers.

Key innovation over OpenMythos:
  OpenMythos = deterministic recursive depth (single trajectory)
  GRAM = stochastic recursive depth (multiple trajectories)
  Combined = depth extrapolation + width scaling

Architecture:
  Input → Encoder → [Prelude → Stochastic Recurrent Block × T → Coda] × N samples → LPRM → Best Output

  Stochastic Recurrent Block:
    1. K low-level refinements via fL (standard transformer)
    2. High-level update fH produces deterministic proposal ut
    3. Learnable stochastic guidance: ht = ut + εt
       where εt ~ N(μ(ht-1, x), σ(ht-1, x))
    4. Mean encodes state-dependent direction
    5. Variance controls exploration amount

Training: Amortized variational inference (ELBO)
  - Posterior q(ε|x,y) — sees both input and answer
  - Prior p(ε|x) — sees only input
  - KL divergence regularizes exploration

Inference: Two scaling axes
  - Depth: more recursive steps (train 16 → test 64)
  - Width: more parallel trajectory samples (N=20-100)
  - LPRM selects best trajectory

Authors: RavenX LLC / @DeadByDawn101
Paper: arXiv:2605.19376 (Baek et al., KAIST/NYU/Mila, Bengio)
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple


class StochasticGuidance(nn.Module):
    """Learned stochastic guidance module.
    
    Produces mean and log-variance for the guidance noise εt,
    conditioned on current state and input embedding.
    
    ht = ut + εt where εt ~ N(μ(ht-1, x), σ(ht-1, x))
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or dim * 2
        # Mean network — state-dependent direction
        self.mean_net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Log-variance network — exploration amount
        self.logvar_net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    
    def __call__(self, h: mx.array, x_embed: mx.array) -> Tuple[mx.array, mx.array]:
        """Returns (mean, log_variance) for guidance noise."""
        combined = mx.concatenate([h, x_embed], axis=-1)
        mean = self.mean_net(combined)
        logvar = mx.clip(self.logvar_net(combined), -10, 2)  # Stability
        return mean, logvar
    
    def sample(self, h: mx.array, x_embed: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Sample guidance noise using reparameterization trick.
        Returns (noise, mean, logvar)."""
        mean, logvar = self(h, x_embed)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape)
        noise = mean + std * eps
        return noise, mean, logvar


class PosteriorGuidance(nn.Module):
    """Posterior q(ε|x,y) — sees both input and target.
    Used during training to provide better guidance signal."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or dim * 2
        self.mean_net = nn.Sequential(
            nn.Linear(dim * 3, hidden),  # h + x_embed + y_embed
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    
    def __call__(self, h: mx.array, x_embed: mx.array, y_embed: mx.array) -> Tuple[mx.array, mx.array]:
        combined = mx.concatenate([h, x_embed, y_embed], axis=-1)
        mean = self.mean_net(combined)
        logvar = mx.clip(self.logvar_net(combined), -10, 2)
        return mean, logvar
    
    def sample(self, h: mx.array, x_embed: mx.array, y_embed: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        mean, logvar = self(h, x_embed, y_embed)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape)
        noise = mean + std * eps
        return noise, mean, logvar


class TransformerBlock(nn.Module):
    """Standard transformer block for low-level refinement."""
    
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff1 = nn.Linear(dim, dim * 4, bias=False)
        self.ff2 = nn.Linear(dim * 4, dim, bias=False)
        self.heads = heads
        self.hd = dim // heads
    
    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, self.hd)
        q = qkv[:,:,0].transpose(0,2,1,3)
        k = qkv[:,:,1].transpose(0,2,1,3)
        v = qkv[:,:,2].transpose(0,2,1,3)
        s = (q @ k.transpose(0,1,3,2)) * (self.hd ** -0.5)
        mask = mx.triu(mx.full((1,1,T,T), -1e9), k=1)
        a = mx.softmax(s + mask, axis=-1)
        out = (a @ v).transpose(0,2,1,3).reshape(B, T, D)
        x = x + self.out(out)
        x = x + self.ff2(nn.silu(self.ff1(self.norm2(x))))
        return x


class LowLevelRefinement(nn.Module):
    """fL: K steps of transformer refinement within each high-level step."""
    
    def __init__(self, dim: int, heads: int, K: int = 2):
        super().__init__()
        self.blocks = [TransformerBlock(dim, heads) for _ in range(K)]
    
    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return x


class HighLevelUpdate(nn.Module):
    """fH: Produces deterministic proposal ut from current state."""
    
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.block = TransformerBlock(dim, heads)
        self.norm = nn.RMSNorm(dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.block(x))


class LatentProcessRewardModel(nn.Module):
    """LPRM: Predicts output correctness from latent state.
    Used at inference to select best trajectory from N samples."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
    
    def __call__(self, h: mx.array) -> mx.array:
        """Returns reward score for each position."""
        return self.net(h)  # (B, T, 1)
    
    def score_trajectory(self, h: mx.array) -> mx.array:
        """Returns single score per trajectory (mean over positions)."""
        return mx.mean(self.net(h), axis=(1, 2))  # (B,)


class GRAM(nn.Module):
    """Generative Recursive reAsoning Model.
    
    Combines recursive depth (shared weights looped T times) with
    stochastic width (N parallel trajectory samples) for multi-trajectory
    reasoning with inference-time scaling.
    
    Args:
        vocab_size: Vocabulary size
        dim: Hidden dimension
        heads: Number of attention heads
        K_low: Number of low-level refinement steps per high-level step
        prelude_layers: Number of prelude transformer layers
        coda_layers: Number of coda transformer layers
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 256,
        heads: int = 4,
        K_low: int = 2,
        prelude_layers: int = 1,
        coda_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        
        # Prelude (non-recursive)
        self.prelude = [TransformerBlock(dim, heads) for _ in range(prelude_layers)]
        
        # Recursive components
        self.low_level = LowLevelRefinement(dim, heads, K=K_low)
        self.high_level = HighLevelUpdate(dim, heads)
        
        # Stochastic guidance
        self.prior = StochasticGuidance(dim)         # p(ε|x)
        self.posterior = PosteriorGuidance(dim)       # q(ε|x,y) — training only
        
        # Coda (non-recursive)
        self.coda = [TransformerBlock(dim, heads) for _ in range(coda_layers)]
        
        # Output
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # LPRM for trajectory selection
        self.lprm = LatentProcessRewardModel(dim)
    
    def single_trajectory(
        self,
        ids: mx.array,
        n_steps: int = 16,
        y_embed: Optional[mx.array] = None,
        use_posterior: bool = False,
    ) -> Tuple[mx.array, mx.array, list]:
        """Run a single stochastic trajectory.
        
        Returns: (logits, final_state, kl_terms)
        """
        x = self.embed(ids)
        
        # Prelude
        for block in self.prelude:
            x = block(x)
        
        x_embed = mx.stop_gradient(x)  # Frozen input for guidance conditioning
        kl_terms = []
        
        # Recursive stochastic loop
        for t in range(n_steps):
            # Detach history for stable training (our stop_gradient trick!)
            if t < n_steps - 1:
                x = mx.stop_gradient(x)
            
            # Low-level refinement
            x = self.low_level(x)
            
            # High-level deterministic proposal
            ut = self.high_level(x)
            
            # Stochastic guidance
            if use_posterior and y_embed is not None:
                # Training: sample from posterior q(ε|x,y)
                noise, q_mean, q_logvar = self.posterior.sample(ut, x_embed, y_embed)
                # Also compute prior for KL
                p_mean, p_logvar = self.prior(ut, x_embed)
                # KL(q||p) per step
                kl = 0.5 * mx.sum(
                    p_logvar - q_logvar - 1 +
                    mx.exp(q_logvar - p_logvar) +
                    (q_mean - p_mean)**2 * mx.exp(-p_logvar),
                    axis=-1
                )
                kl_terms.append(mx.mean(kl))
            else:
                # Inference: sample from prior p(ε|x)
                noise, _, _ = self.prior.sample(ut, x_embed)
            
            # Apply stochastic guidance
            x = ut + noise * 0.1  # Scale noise to prevent explosion
        
        # Coda
        for block in self.coda:
            x = block(x)
        
        logits = self.head(self.norm(x))
        return logits, x, kl_terms
    
    def __call__(
        self,
        ids: mx.array,
        n_steps: int = 16,
        y_embed: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Training forward pass: single trajectory with posterior guidance.
        
        Returns: (logits, kl_loss)
        """
        logits, state, kl_terms = self.single_trajectory(
            ids, n_steps, y_embed, use_posterior=(y_embed is not None)
        )
        kl_loss = mx.mean(mx.stack(kl_terms)) if kl_terms else mx.array(0.0)
        return logits, kl_loss
    
    def sample_trajectories(
        self,
        ids: mx.array,
        n_steps: int = 16,
        n_samples: int = 20,
    ) -> Tuple[mx.array, mx.array]:
        """Inference: sample N parallel trajectories, select best via LPRM.
        
        Returns: (best_logits, all_scores)
        """
        all_logits = []
        all_scores = []
        
        for _ in range(n_samples):
            logits, state, _ = self.single_trajectory(ids, n_steps)
            score = self.lprm.score_trajectory(state)
            all_logits.append(logits)
            all_scores.append(score)
        
        # Stack and select best trajectory per batch item
        scores = mx.stack(all_scores, axis=0)  # (N, B)
        best_idx = mx.argmax(scores, axis=0)    # (B,)
        
        # Select best logits
        stacked = mx.stack(all_logits, axis=0)  # (N, B, T, V)
        # For simplicity, take the trajectory with highest mean score
        best_sample_idx = mx.argmax(mx.mean(scores, axis=1))
        best_logits = all_logits[best_sample_idx.item()]
        
        return best_logits, scores


def gram_small():
    """Small GRAM config for testing (comparable to OpenMythos small)."""
    return GRAM(
        vocab_size=32000, dim=256, heads=4,
        K_low=2, prelude_layers=1, coda_layers=1,
    )


def gram_base():
    """Base GRAM config (~50M params)."""
    return GRAM(
        vocab_size=32000, dim=512, heads=8,
        K_low=2, prelude_layers=2, coda_layers=2,
    )


def gram_large():
    """Large GRAM config (~200M params)."""
    return GRAM(
        vocab_size=32000, dim=768, heads=12,
        K_low=3, prelude_layers=3, coda_layers=3,
    )


if __name__ == "__main__":
    print("="*60)
    print("  GRAM-MLX: Generative Recursive Reasoning on Apple Silicon")
    print("="*60)
    
    model = gram_small()
    flat = nn.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in flat)
    print(f"\nGRAM Small: {total:,} parameters")
    
    # Test forward pass
    ids = mx.random.randint(0, 32000, (1, 32))
    
    print(f"\nSingle trajectory (16 steps):")
    logits, kl = model(ids, n_steps=16)
    print(f"  Logits: {logits.shape}")
    print(f"  KL loss: {kl.item():.4f}")
    
    print(f"\nMulti-trajectory (16 steps × 5 samples):")
    best_logits, scores = model.sample_trajectories(ids, n_steps=16, n_samples=5)
    print(f"  Best logits: {best_logits.shape}")
    print(f"  Trajectory scores: {scores.shape}")
    
    # Test depth extrapolation
    print(f"\nDepth extrapolation test:")
    loss_fn = nn.losses.cross_entropy
    for n in [4, 8, 16, 32]:
        logits, _ = model(ids[:, :-1], n_steps=n)
        loss = mx.mean(loss_fn(logits, ids[:, 1:], reduction='none'))
        print(f"  n_steps={n:2d}: loss={loss.item():.4f}")
    
    print(f"\nGRAM-MLX: WORKING!")
    print(f"Next: Train on security data + combine with OpenMythos depth extrapolation")
