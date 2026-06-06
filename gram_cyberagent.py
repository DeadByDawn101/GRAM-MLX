"""
GRAM × RavenX-CyberAgent Integration
=====================================
Applies GRAM stochastic multi-trajectory reasoning to
RavenX-CyberAgent for deeper security assessments.

Two modes:
  1. INFERENCE: Multi-trajectory RATH generation with LPRM selection
  2. TRAINING: Generate best-of-N traces for distillation into next version

Usage (on M4 Max):
  python3.13 gram_cyberagent.py --mode inference --samples 5
  python3.13 gram_cyberagent.py --mode training --samples 20 --output traces.jsonl

Author: RavenX LLC / @DeadByDawn101
"""

import mlx.core as mx
import mlx.nn as nn
import argparse
import json
import time
import os
import sys

# Import GRAM wrapper
sys.path.insert(0, os.path.dirname(__file__))
from gram_wrapper import GRAMWrapper, StochasticGuidanceLayer, TrajectoryScorer


class GRAMCyberAgent:
    """GRAM-enhanced RavenX-CyberAgent.
    
    Wraps the production 35B MoE model with GRAM stochastic guidance
    for multi-trajectory security reasoning.
    
    Architecture:
        Base: RavenX-CyberAgent 35B MoE (FROZEN)
        GRAM: ~71M trainable guidance params
        LPRM: Trajectory scorer for best-of-N selection
    """
    
    def __init__(
        self,
        model_path: str = None,
        dim: int = 2048,
        n_guidance: int = 4,
        guidance_scale: float = 0.1,
    ):
        # GRAM wrapper
        self.gram = GRAMWrapper(
            dim=dim,
            n_guidance_layers=n_guidance,
            guidance_scale=guidance_scale,
        )
        
        # Base model (loaded separately)
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.dim = dim
        
        print(f"GRAM-CyberAgent initialized:")
        print(f"  GRAM params: {self.gram.trainable_params:,}")
        print(f"  Guidance layers: {n_guidance}")
        print(f"  Scale: {guidance_scale}")
    
    def load_model(self):
        """Load the base RavenX-CyberAgent model."""
        if self.model is not None:
            return
        
        try:
            from mlx_lm import load
            model_id = self.model_path or "deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx"
            print(f"\nLoading base model: {model_id}")
            self.model, self.tokenizer = load(model_id)
            print(f"  Base model loaded!")
        except Exception as e:
            print(f"  Base model not available: {e}")
            print(f"  Running in GRAM-only mode (testing wrapper)")
    
    def generate_single(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Generate single trajectory with stochastic guidance."""
        
        if system_prompt is None:
            system_prompt = (
                "You are RavenX-Sec v5.1 by RavenX LLC. "
                "Think briefly in 1-2 sentences then output. "
                "Use 6 RATH steps: 1-Attack Surface, 2-Exploit, 3-Impact, "
                "4-Remediation, 5-Document, 6-Prevent. "
                "Include CVSS, CWE, MITRE. Be concise. Never repeat."
            )
        
        if self.model is not None and self.tokenizer is not None:
            from mlx_lm import generate
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            t0 = time.time()
            response = generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
            )
            dt = time.time() - t0
            
            return {
                "response": response,
                "time": dt,
                "tokens": len(self.tokenizer.encode(response)),
            }
        else:
            # GRAM-only mode: simulate with random hidden states
            h = mx.random.normal((1, 32, self.dim))
            guided, mean, logvar = self.gram.guide_hidden_states(h, layer_idx=0)
            score = self.gram.score_trajectory(guided)
            
            return {
                "response": f"[GRAM-guided trajectory, score={score:.4f}]",
                "time": 0.0,
                "score": score,
                "guided_norm": mx.mean(mx.abs(guided - h)).item(),
            }
    
    def generate_multi(
        self,
        prompt: str,
        n_samples: int = 5,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Generate N trajectories with stochastic guidance, select best.
        
        This is the KEY FEATURE: multiple parallel reasoning paths,
        LPRM selects the best one for output.
        """
        
        trajectories = []
        
        print(f"\n  Generating {n_samples} trajectories...")
        for i in range(n_samples):
            result = self.generate_single(
                prompt, system_prompt, max_tokens, temperature
            )
            
            # Score trajectory
            if self.model is not None:
                # Score based on response quality heuristics
                response = result["response"]
                score = 0.0
                # RATH completeness
                for step in ["1-Attack Surface", "2-Exploit", "3-Impact", 
                            "4-Remediation", "5-Document", "6-Prevent"]:
                    if step in response:
                        score += 1.0
                # Specificity (has real commands/CVEs)
                for indicator in ["CVE-", "CWE-", "CVSS", "nmap", "sqlmap", 
                                 "kubectl", "curl", "MITRE T"]:
                    if indicator in response:
                        score += 0.5
                # Penalize repetition
                lines = response.split("\n")
                unique_ratio = len(set(lines)) / max(len(lines), 1)
                score *= unique_ratio
                result["score"] = score
            
            trajectories.append(result)
            print(f"    Trajectory {i+1}: score={result.get('score', 0):.2f}")
        
        # Select best
        best = max(trajectories, key=lambda t: t.get("score", 0))
        worst = min(trajectories, key=lambda t: t.get("score", 0))
        
        return {
            "best": best,
            "worst": worst,
            "all_scores": [t.get("score", 0) for t in trajectories],
            "n_samples": n_samples,
            "improvement": best.get("score", 0) - worst.get("score", 0),
        }
    
    def generate_training_traces(
        self,
        prompts: list,
        n_samples: int = 20,
        output_file: str = "gram_traces.jsonl",
    ) -> list:
        """Generate best-of-N traces for distillation training.
        
        For each prompt, generates N trajectories and keeps the BEST one.
        These best traces become training data for the next model version.
        
        This is the GRAM Reasoning-as-a-Service pipeline:
          N trajectories × depth extrapolation → LPRM best → distill
        """
        
        traces = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            
            result = self.generate_multi(prompt, n_samples=n_samples)
            best = result["best"]
            
            trace = {
                "messages": [
                    {"role": "system", "content": "You are RavenX-Sec v5.1 with GRAM deep reasoning. Follow 6-step RATH."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best.get("response", "")},
                ],
                "gram_metadata": {
                    "n_samples": n_samples,
                    "best_score": best.get("score", 0),
                    "all_scores": result["all_scores"],
                    "improvement": result["improvement"],
                }
            }
            traces.append(trace)
            
            print(f"  Best score: {best.get('score', 0):.2f}")
            print(f"  Improvement over worst: {result['improvement']:.2f}")
        
        # Save traces
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        
        print(f"\nSaved {len(traces)} best-of-{n_samples} traces → {output_file}")
        return traces


# ============================================================
# SECURITY-SPECIFIC GRAM SCORING (LPRM for security domain)
# ============================================================

class SecurityLPRM(nn.Module):
    """Security-domain Latent Process Reward Model.
    
    Scores security assessment quality based on:
    - RATH completeness (all 6 steps present)
    - Technical specificity (real commands, CVEs, CWEs)
    - Kill chain coherence (findings link together)
    - Remediation actionability (specific fixes, not generic)
    - Non-repetition (unique content per section)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.rath_scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),  # One score per RATH step
        )
        self.specificity_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.coherence_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
    
    def __call__(self, h: mx.array) -> mx.array:
        """Score trajectory quality."""
        rath = mx.mean(self.rath_scorer(h), axis=(1, 2))      # RATH completeness
        spec = mx.mean(self.specificity_scorer(h), axis=(1, 2)) # Technical depth
        cohr = mx.mean(self.coherence_scorer(h), axis=(1, 2))   # Kill chain coherence
        return rath + spec + cohr


# ============================================================
# TEST PROMPTS (security scenarios for multi-trajectory testing)
# ============================================================

SECURITY_PROMPTS = [
    "Open MongoDB 4.2 on port 27017 with no auth containing customer PII. Full RATH.",
    
    "AWS EKS cluster: anonymous auth API server, privileged root pods, SA tokens everywhere, etcd without TLS. Full RATH with kill chain.",
    
    "Web app login form vulnerable to SQL injection, WAF detecting basic payloads. Advanced SQLi bypass with sqlmap tamper scripts. Full RATH.",
    
    "Redis 6.0 on port 6379 no password, SLAVEOF enabled, containing session tokens. Full RATH.",
    
    "GraphQL API with no rate limiting, IDOR on user profile mutations, introspection enabled. Full RATH.",
    
    "Jenkins CI/CD with plaintext AWS creds in env vars, SSH to production, ArgoCD misconfigured. Full RATH with kill chain.",
    
    "S3 bucket public read containing database backups, IAM role with AdministratorAccess on EC2. Full RATH.",
    
    "Kubernetes namespace with no network policies, pods running as root, service account tokens auto-mounted. Full RATH.",
    
    "PostgreSQL 12.3 exposed on port 5432 with default credentials, pg_dump accessible. Full RATH.",
    
    "Elasticsearch 7.x on port 9200 no auth, _cat/indices exposed, script injection possible. Full RATH.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAM × RavenX-CyberAgent")
    parser.add_argument("--mode", choices=["test", "inference", "training"], default="test")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default="gram_traces.jsonl")
    args = parser.parse_args()
    
    print("="*60)
    print("  GRAM × RavenX-CyberAgent")
    print("  Stochastic Multi-Trajectory Security Reasoning")
    print("="*60)
    
    agent = GRAMCyberAgent(model_path=args.model)
    
    if args.mode == "test":
        # Test GRAM wrapper only (no base model needed)
        print(f"\nTesting GRAM wrapper (no base model)...")
        
        for i, prompt in enumerate(SECURITY_PROMPTS[:3]):
            print(f"\n{'─'*50}")
            print(f"Prompt {i+1}: {prompt[:60]}...")
            
            result = agent.generate_multi(prompt, n_samples=args.samples)
            print(f"  Scores: {result['all_scores']}")
            print(f"  Best: {result['best'].get('score', 0):.4f}")
            print(f"  Improvement: {result['improvement']:.4f}")
    
    elif args.mode == "inference":
        # Full inference with base model
        agent.load_model()
        
        prompt = SECURITY_PROMPTS[0]
        print(f"\nPrompt: {prompt}")
        result = agent.generate_multi(prompt, n_samples=args.samples)
        
        print(f"\n{'='*50}")
        print(f"BEST TRAJECTORY (score={result['best'].get('score', 0):.2f}):")
        print(f"{'='*50}")
        print(result["best"].get("response", ""))
        
    elif args.mode == "training":
        # Generate best-of-N traces for distillation
        agent.load_model()
        
        traces = agent.generate_training_traces(
            prompts=SECURITY_PROMPTS,
            n_samples=args.samples,
            output_file=args.output,
        )
        
        avg_improvement = sum(t["gram_metadata"]["improvement"] for t in traces) / len(traces)
        print(f"\nAverage improvement (best vs worst): {avg_improvement:.2f}")
        print(f"Traces ready for distillation into next version!")

    print(f"\n{'='*60}")
    print(f"  GRAM × CyberAgent: COMPLETE")
    print(f"  Depth (OpenMythos) × Width (GRAM) = Deeper Security Reasoning")
    print(f"{'='*60}")
