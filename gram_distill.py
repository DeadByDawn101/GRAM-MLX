"""
GRAM Distillation — Bake multi-trajectory reasoning INTO the model.

Same approach as OpenMythos depth distillation:
  OpenMythos: Train at 2 loops → generate at 8 → distill best traces
  GRAM:      Generate N trajectories → score → distill BEST-of-N traces

The model learns to produce GRAM-quality output in a SINGLE pass.
Combined with self-improving agent data = model that learns + reasons wide.

Pipeline:
  1. Load RavenX-CyberAgent 35B
  2. For each security prompt, generate N trajectories
  3. Score each trajectory (RATH completeness, specificity, coherence)
  4. Keep BEST trajectory as training data
  5. Save as JSONL for next training round
  6. Fine-tune → model produces GRAM-quality output natively

Usage:
  python3.13 gram_distill.py --samples 10 --prompts 50
  python3.13 gram_distill.py --samples 20 --prompts 100 --output gram_traces_r10.jsonl

Author: RavenX LLC / @DeadByDawn101
"""

import json
import time
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def score_rath(response: str) -> dict:
    """Score a RATH assessment on multiple dimensions."""
    scores = {
        "rath_steps": 0,
        "specificity": 0,
        "commands": 0,
        "compliance": 0,
        "uniqueness": 0,
        "total": 0,
    }
    
    # RATH completeness (0-6)
    for step in ["1-Attack Surface", "2-Exploit", "3-Impact",
                 "4-Remediation", "5-Document", "6-Prevent"]:
        if step in response:
            scores["rath_steps"] += 1
    
    # Technical specificity (0-10)
    specifics = ["CVE-", "CWE-", "CVSS", "MITRE T", "OWASP A",
                 "nmap", "sqlmap", "kubectl", "curl", "redis-cli",
                 "mongodump", "nuclei", "ffuf", "subfinder", "burp"]
    for s in specifics:
        if s in response:
            scores["specificity"] += 0.67
    
    # Real commands (0-5)
    command_indicators = ["```", "$ ", "--host", "--port", "-u ",
                         "SELECT ", "INSERT ", "DROP ", "|", "grep",
                         ".find(", ".exec(", "config set"]
    for c in command_indicators:
        if c in response:
            scores["commands"] += 0.38
    
    # Compliance references (0-3)
    compliance = ["NIST", "ISO 27001", "PCI DSS", "GDPR", "SOC 2",
                  "HIPAA", "CCPA", "OWASP"]
    for c in compliance:
        if c in response:
            scores["compliance"] += 0.375
    
    # Uniqueness (penalize repetition) (0-1)
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if lines:
        scores["uniqueness"] = len(set(lines)) / len(lines)
    
    # Total
    scores["total"] = (
        scores["rath_steps"] * 2 +      # 0-12
        scores["specificity"] +           # 0-10
        scores["commands"] +              # 0-5
        scores["compliance"] +            # 0-3
        scores["uniqueness"] * 5          # 0-5
    )  # max ~35
    
    return scores


# Diverse security prompts covering many attack surfaces
SECURITY_PROMPTS = [
    # Web Application
    "Login form SQL injection with WAF bypass. PostgreSQL backend, Express.js API. Full RATH.",
    "Stored XSS in user profile metadata field. React frontend, CSP headers misconfigured. Full RATH.",
    "SSRF via webhook URL parameter. Internal AWS metadata service accessible. Full RATH.",
    "JWT authentication bypass via alg:none attack. Auth0 integration. Full RATH.",
    "GraphQL API with introspection enabled, no rate limiting, IDOR on mutations. Full RATH.",
    "CSRF on money transfer endpoint. No SameSite cookies, no CSRF token. Full RATH.",
    "Prototype pollution via lodash merge in Node.js API. Full RATH.",
    "Insecure deserialization in Java Spring Boot application. Full RATH.",
    "Path traversal via file upload endpoint. No extension validation. Full RATH.",
    "Race condition in coupon redemption API. No mutex, no idempotency key. Full RATH.",
    
    # Cloud Infrastructure
    "AWS S3 bucket public read with database backups containing PII. Full RATH.",
    "IAM role with AdministratorAccess attached to EC2 instance. Full RATH.",
    "AWS Lambda function with hardcoded API keys in environment variables. Full RATH.",
    "CloudFront distribution serving content from misconfigured origin. Full RATH.",
    "RDS instance publicly accessible with default credentials. Full RATH.",
    
    # Container / Kubernetes
    "Kubernetes API server with anonymous authentication enabled. Full RATH.",
    "Privileged pods running as root with host PID namespace. Full RATH.",
    "Kubernetes secrets stored as base64 containing SSH private keys. Full RATH.",
    "No network policies between namespaces in EKS cluster. Full RATH.",
    "etcd accessible without TLS on port 2379. Full RATH with kill chain.",
    
    # Database
    "MongoDB 4.2 on port 27017 no auth, customer PII in collections. Full RATH.",
    "Redis 6.0 port 6379 no password, SLAVEOF enabled, session tokens. Full RATH.",
    "Elasticsearch 7.x on port 9200 no auth, /_cat/indices exposed. Full RATH.",
    "PostgreSQL 12.3 on port 5432 default creds, pg_dump accessible. Full RATH.",
    "MySQL 8.0 with root:root, binary logging to world-readable files. Full RATH.",
    
    # CI/CD
    "Jenkins CI/CD with plaintext AWS creds in env vars, SSH to prod. Full RATH.",
    "GitHub Actions workflow injection via repository_dispatch event. Full RATH.",
    "ArgoCD API exposed without authentication, direct Kubernetes access. Full RATH.",
    "GitLab CI runner with privileged Docker executor, shared runners. Full RATH.",
    "CircleCI with SSH keys stored as project env vars, no rotation. Full RATH.",
    
    # Network
    "Open VPN server with default certificates, split tunneling disabled. Full RATH.",
    "DNS zone transfer enabled on primary nameserver. Full RATH.",
    "SNMP v2c with public community string on network devices. Full RATH.",
    "NFS exports with no_root_squash to 0.0.0.0/0. Full RATH.",
    "FTP server with anonymous login, write access to web root. Full RATH.",
    
    # Crypto / Auth
    "OAuth 2.0 redirect_uri validation bypass via open redirect. Full RATH.",
    "SAML assertion signature bypass via XML signature wrapping. Full RATH.",
    "Password reset token predictable via sequential generation. Full RATH.",
    "API key rotation not enforced, keys valid indefinitely. Full RATH.",
    "2FA bypass via response manipulation, TOTP seed not encrypted. Full RATH.",
    
    # Advanced / Kill Chain
    "Full external pentest: React + Node.js + PostgreSQL + AWS EKS + Auth0 + Redis. Map complete kill chain from initial access to data exfiltration.",
    "Internal network pentest: Windows AD environment, Kerberoasting, DCSync, lateral movement via PSExec. Full kill chain RATH.",
    "Cloud pentest: Multi-account AWS with cross-account role assumption, S3 exfil, Lambda persistence. Kill chain RATH.",
    "Supply chain attack via compromised npm package in CI/CD pipeline. Full RATH with kill chain.",
    "API security assessment: REST + GraphQL + WebSocket endpoints with OAuth2 + JWT. Full RATH.",
    
    # Bug Bounty
    "Bug bounty: subdomain takeover via dangling CNAME to decommissioned S3 bucket. Full RATH.",
    "Bug bounty: account takeover via password reset link leakage in referrer header. Full RATH.",
    "Bug bounty: privilege escalation from user to admin via mass assignment on PUT /api/users/:id. Full RATH.",
    "Bug bounty: information disclosure via verbose error messages exposing stack traces and file paths. Full RATH.",
    "Bug bounty: blind SSRF via PDF generation feature accessing internal metadata service. Full RATH.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAM Distillation — bake multi-trajectory reasoning into the model")
    parser.add_argument("--samples", type=int, default=5, help="Trajectories per prompt (best-of-N)")
    parser.add_argument("--prompts", type=int, default=10, help="Number of prompts to process")
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--output", type=str, default="gram_distill_traces.jsonl")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    
    print("="*60)
    print("  GRAM DISTILLATION — Bake Multi-Trajectory Into Model")
    print("="*60)
    print(f"  Samples per prompt: {args.samples}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Output: {args.output}")
    
    # Load model
    try:
        from mlx_lm import load, generate
        model_path = args.model or "deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx"
        print(f"\nLoading: {model_path}")
        model, tokenizer = load(model_path)
        print("  Model loaded!")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Run with: --model ~/Developer/RavenX-Sec/models/checkpoints/ravenx-sec-v5.0-fused")
        sys.exit(1)
    
    system_prompt = (
        "You are RavenX-Sec v5.1, a 35B autonomous security agent by RavenX LLC. "
        "Think briefly in 1-2 sentences then output. "
        "ALWAYS use EXACT 6 RATH step names: 1-Attack Surface, 2-Exploit, 3-Impact, "
        "4-Remediation, 5-Document, 6-Prevent. "
        "Include CVSS 3.1 scores, CWE IDs, MITRE ATT&CK TTPs, and compliance mapping. "
        "Provide REAL commands. Be concise and direct. Never repeat."
    )
    
    prompts = SECURITY_PROMPTS[:args.prompts]
    traces = []
    total_improvement = 0
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*60}")
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:70]}...")
        
        trajectories = []
        
        for s in range(args.samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            t0 = time.time()
            response = generate(model, tokenizer, prompt=formatted, max_tokens=args.max_tokens)
            dt = time.time() - t0
            
            scores = score_rath(response)
            trajectories.append({
                "response": response,
                "scores": scores,
                "time": dt,
            })
            print(f"  Trajectory {s+1}: total={scores['total']:.1f} "
                  f"(rath={scores['rath_steps']}, spec={scores['specificity']:.1f}, "
                  f"cmd={scores['commands']:.1f}) [{dt:.1f}s]")
        
        # Select BEST
        best = max(trajectories, key=lambda t: t["scores"]["total"])
        worst = min(trajectories, key=lambda t: t["scores"]["total"])
        improvement = best["scores"]["total"] - worst["scores"]["total"]
        total_improvement += improvement
        
        print(f"  BEST:  {best['scores']['total']:.1f}")
        print(f"  WORST: {worst['scores']['total']:.1f}")
        print(f"  IMPROVEMENT: +{improvement:.1f}")
        
        # Save best as training data
        trace = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": best["response"]},
            ],
            "gram_metadata": {
                "n_samples": args.samples,
                "best_score": best["scores"]["total"],
                "worst_score": worst["scores"]["total"],
                "improvement": improvement,
                "scores_breakdown": best["scores"],
            }
        }
        traces.append(trace)
    
    # Save traces
    with open(args.output, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    
    avg_improvement = total_improvement / len(prompts) if prompts else 0
    
    print(f"\n{'='*60}")
    print(f"  GRAM DISTILLATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Prompts processed: {len(prompts)}")
    print(f"  Samples per prompt: {args.samples}")
    print(f"  Total generations: {len(prompts) * args.samples}")
    print(f"  Average improvement (best vs worst): +{avg_improvement:.1f}")
    print(f"  Traces saved: {args.output}")
    print(f"")
    print(f"  Next: Add traces to training data")
    print(f"    cat {args.output} >> ~/Developer/RavenX-Sec/data/train.jsonl")
    print(f"    Then train Round 10!")
    print(f"{'='*60}")
