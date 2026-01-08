#!/usr/bin/env python3
"""
Generate PeakWeights analysis for top 25 LLM models.

Usage:
    python scripts/generate_top25.py --output results/top25/
    python scripts/generate_top25.py --model deepseek-ai/DeepSeek-R1 --k 50
    python scripts/generate_top25.py --list  # Show all models
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Top 25 LLM Models for 2026
# Ordered by priority (most important first)
TOP_25_MODELS = [
    # Tier 1: Frontier Open Models (require significant compute)
    {
        "id": "deepseek-v3",
        "huggingface_id": "deepseek-ai/DeepSeek-V3",
        "name": "DeepSeek-V3",
        "organization": "DeepSeek",
        "parameters": "671B (37B active)",
        "architecture": "MoE",
        "min_gpu_memory": "160GB",  # Multi-GPU required
    },
    {
        "id": "deepseek-r1",
        "huggingface_id": "deepseek-ai/DeepSeek-R1",
        "name": "DeepSeek-R1",
        "organization": "DeepSeek",
        "parameters": "671B (37B active)",
        "architecture": "MoE",
        "min_gpu_memory": "160GB",
    },
    {
        "id": "qwen3-235b",
        "huggingface_id": "Qwen/Qwen3-235B-A22B",
        "name": "Qwen3-235B-A22B",
        "organization": "Alibaba",
        "parameters": "235B (22B active)",
        "architecture": "MoE",
        "min_gpu_memory": "160GB",
    },
    {
        "id": "llama-4-maverick",
        "huggingface_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "name": "Llama 4 Maverick",
        "organization": "Meta",
        "parameters": "400B (17B active)",
        "architecture": "MoE",
        "min_gpu_memory": "160GB",
    },

    # Tier 2: High-Performance 70B Class (require ~140GB VRAM)
    {
        "id": "qwen2.5-72b",
        "huggingface_id": "Qwen/Qwen2.5-72B-Instruct",
        "name": "Qwen2.5-72B-Instruct",
        "organization": "Alibaba",
        "parameters": "72B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "140GB",
    },
    {
        "id": "llama-3.3-70b",
        "huggingface_id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama-3.3-70B-Instruct",
        "organization": "Meta",
        "parameters": "70B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "140GB",
    },
    {
        "id": "deepseek-r1-distill-llama-70b",
        "huggingface_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "name": "DeepSeek-R1-Distill-Llama-70B",
        "organization": "DeepSeek",
        "parameters": "70B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "140GB",
    },
    {
        "id": "mixtral-8x22b",
        "huggingface_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "name": "Mixtral-8x22B-Instruct-v0.1",
        "organization": "Mistral AI",
        "parameters": "141B (39B active)",
        "architecture": "MoE",
        "min_gpu_memory": "100GB",
    },

    # Tier 3: Efficient 7B-32B Class (runnable on single GPU)
    {
        "id": "qwen2.5-coder-32b",
        "huggingface_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "name": "Qwen2.5-Coder-32B-Instruct",
        "organization": "Alibaba",
        "parameters": "32B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "64GB",
    },
    {
        "id": "gemma-2-27b",
        "huggingface_id": "google/gemma-2-27b-it",
        "name": "Gemma-2-27B-it",
        "organization": "Google",
        "parameters": "27B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "54GB",
    },
    {
        "id": "codestral-22b",
        "huggingface_id": "mistralai/Codestral-22B-v0.1",
        "name": "Codestral-22B-v0.1",
        "organization": "Mistral AI",
        "parameters": "22B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "44GB",
    },
    {
        "id": "qwen2.5-14b",
        "huggingface_id": "Qwen/Qwen2.5-14B-Instruct",
        "name": "Qwen2.5-14B-Instruct",
        "organization": "Alibaba",
        "parameters": "14B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "28GB",
    },
    {
        "id": "phi-4",
        "huggingface_id": "microsoft/phi-4",
        "name": "Phi-4",
        "organization": "Microsoft",
        "parameters": "14B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "28GB",
    },
    {
        "id": "olmo-2-13b",
        "huggingface_id": "allenai/OLMo-2-13B-Instruct",
        "name": "OLMo-2-13B-Instruct",
        "organization": "AI2",
        "parameters": "13B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "26GB",
    },
    {
        "id": "gemma-2-9b",
        "huggingface_id": "google/gemma-2-9b-it",
        "name": "Gemma-2-9B-it",
        "organization": "Google",
        "parameters": "9B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "18GB",
    },

    # Tier 4: 7B Models (most accessible)
    {
        "id": "qwen2.5-7b",
        "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen2.5-7B-Instruct",
        "organization": "Alibaba",
        "parameters": "7B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "14GB",
    },
    {
        "id": "deepseek-r1-distill-qwen-7b",
        "huggingface_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "organization": "DeepSeek",
        "parameters": "7B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "14GB",
    },
    {
        "id": "mistral-7b-v0.3",
        "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral-7B-Instruct-v0.3",
        "organization": "Mistral AI",
        "parameters": "7B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "14GB",
    },

    # Tier 5: Small & Efficient Models (can run on consumer GPUs)
    {
        "id": "phi-3.5-mini",
        "huggingface_id": "microsoft/Phi-3.5-mini-instruct",
        "name": "Phi-3.5-mini-instruct",
        "organization": "Microsoft",
        "parameters": "3.8B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "8GB",
    },
    {
        "id": "qwen2.5-3b",
        "huggingface_id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B-Instruct",
        "organization": "Alibaba",
        "parameters": "3B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "6GB",
    },
    {
        "id": "gemma-2-2b",
        "huggingface_id": "google/gemma-2-2b-it",
        "name": "Gemma-2-2B-it",
        "organization": "Google",
        "parameters": "2B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "4GB",
    },
    {
        "id": "smollm2-1.7b",
        "huggingface_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2-1.7B-Instruct",
        "organization": "HuggingFace",
        "parameters": "1.7B",
        "architecture": "Dense Transformer",
        "min_gpu_memory": "4GB",
    },
]


def list_models():
    """Print all available models."""
    print("\nTop 25 LLM Models for PeakWeights Analysis")
    print("=" * 80)
    for i, model in enumerate(TOP_25_MODELS, 1):
        print(f"{i:2}. {model['name']:<40} {model['parameters']:<20} {model['min_gpu_memory']}")
    print("\nUsage: python scripts/generate_top25.py --model MODEL_ID")
    print("       python scripts/generate_top25.py --all --output results/")


def analyze_model(model_id: str, k: int = 50, output_dir: str = None, device: str = "auto"):
    """Run PeakWeights analysis on a single model."""
    try:
        from peakweights import find, save
    except ImportError:
        print("Error: peakweights not installed. Run: pip install -e .")
        sys.exit(1)

    # Find model info
    model_info = None
    for m in TOP_25_MODELS:
        if m["id"] == model_id or m["huggingface_id"] == model_id:
            model_info = m
            break

    if model_info is None:
        print(f"Error: Model '{model_id}' not found in top 25 list.")
        print("Use --list to see available models, or provide a HuggingFace model ID directly.")
        # Allow analyzing any model
        model_info = {
            "id": model_id.replace("/", "-"),
            "huggingface_id": model_id,
            "name": model_id,
        }

    hf_id = model_info["huggingface_id"]
    print(f"\nAnalyzing: {model_info['name']}")
    print(f"HuggingFace ID: {hf_id}")
    print(f"K: {k}")
    print("-" * 60)

    # Run analysis
    try:
        critical = find(hf_id, k=k, device=device)
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return None

    # Prepare results
    results = {
        "model_id": model_info["id"],
        "huggingface_id": hf_id,
        "name": model_info["name"],
        "k": k,
        "analysis_date": datetime.now().isoformat(),
        "critical_weights": [
            {
                "rank": i + 1,
                "score": w.score,
                "module": w.module,
                "row": w.row,
                "col": w.col,
            }
            for i, w in enumerate(critical)
        ],
    }

    # Calculate statistics
    scores = [w.score for w in critical]
    results["stats"] = {
        "top_score": scores[0] if scores else 0,
        "min_score": scores[-1] if scores else 0,
        "total_score": sum(scores),
    }

    # Module distribution
    module_types = {}
    for w in critical[:10]:
        if "mlp" in w.module.lower() or "down_proj" in w.module.lower() or "up_proj" in w.module.lower():
            t = "MLP"
        elif "attn" in w.module.lower() or "attention" in w.module.lower():
            t = "Attention"
        elif "lm_head" in w.module.lower() or "embed" in w.module.lower():
            t = "lm_head"
        else:
            t = "Other"
        module_types[t] = module_types.get(t, 0) + 1
    results["module_distribution"] = module_types

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save .pwi file
        pwi_path = output_path / f"{model_info['id']}.pwi"
        save(critical, str(pwi_path), model_name=hf_id)
        print(f"Saved: {pwi_path}")

        # Save JSON summary
        json_path = output_path / f"{model_info['id']}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {json_path}")

    return results


def analyze_all(output_dir: str, k: int = 50, device: str = "auto", tier: int = None):
    """Analyze all models (or specific tier)."""
    results = []

    # Filter by tier if specified
    models = TOP_25_MODELS
    if tier:
        tier_ranges = {
            1: (0, 4),    # Frontier
            2: (4, 8),    # 70B class
            3: (8, 15),   # 7B-32B class
            4: (15, 18),  # 7B models
            5: (18, 22),  # Small models
        }
        if tier in tier_ranges:
            start, end = tier_ranges[tier]
            models = TOP_25_MODELS[start:end]

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing {model['name']}...")
        try:
            result = analyze_model(model["id"], k=k, output_dir=output_dir, device=device)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # Save combined results
    if results and output_dir:
        combined_path = Path(output_dir) / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved combined results: {combined_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate PeakWeights analysis for top 25 LLM models"
    )
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--model", type=str, help="Model ID to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all models")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Analyze specific tier only")
    parser.add_argument("--k", type=int, default=50, help="Number of top weights to find (default: 50)")
    parser.add_argument("--output", type=str, default="results/top25", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, mps, or auto")

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.model:
        analyze_model(args.model, k=args.k, output_dir=args.output, device=args.device)
    elif args.all:
        analyze_all(args.output, k=args.k, device=args.device, tier=args.tier)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
