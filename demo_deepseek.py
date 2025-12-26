#!/usr/bin/env python3
"""
PeakWeights Demo: Find critical weights in DeepSeek models.

Usage:
    # For DeepSeek-R1 Distill (smaller, fits on consumer GPUs)
    python demo_deepseek.py deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

    # For DeepSeek-R1 Distill 14B
    python demo_deepseek.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

    # For full DeepSeek-R1 (requires 8x H100 80GB or similar)
    python demo_deepseek.py deepseek-ai/DeepSeek-R1 --dtype bfloat16

Available DeepSeek Models:
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (~3GB VRAM)
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   (~14GB VRAM)
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  (~28GB VRAM)
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  (~64GB VRAM)
    - deepseek-ai/DeepSeek-R1-Distill-Llama-8B  (~16GB VRAM)
    - deepseek-ai/DeepSeek-R1-Distill-Llama-70B (~140GB VRAM)
    - deepseek-ai/DeepSeek-R1                   (~1.3TB VRAM, 671B MoE)
    - deepseek-ai/DeepSeek-V3                   (~1.3TB VRAM, 671B MoE)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from peakweights.peakweights import PeakWeightsFinder, print_results


def main():
    parser = argparse.ArgumentParser(
        description="PeakWeights Demo for DeepSeek models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model to analyze (default: DeepSeek-R1-Distill-Qwen-1.5B)"
    )
    parser.add_argument("--top_k", "-k", type=int, default=10,
                        help="Number of critical weights to find")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--output", "-o", help="Save results to JSON")

    args = parser.parse_args()

    # Check GPU
    print("\n" + "="*70)
    print("  PEAKWEIGHTS DEMO: DeepSeek Critical Weight Analysis")
    print("="*70)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name}")
        print(f"  Memory: {gpu_mem:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected, running on CPU (very slow)")

    print(f"\n  Model: {args.model}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Dtype: {args.dtype}")
    print("="*70 + "\n")

    # Map dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Run analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"

    finder = PeakWeightsFinder(
        args.model,
        top_k=args.top_k,
        device=device,
        dtype=dtype_map[args.dtype],
        include_moe_router=True,  # Important for MoE models like DeepSeek
    )

    results = finder.find()

    # Print results
    print_results(results, args.model, finder.total_params)

    # Analyze where critical weights appear
    print("\n" + "="*70)
    print("  ANALYSIS: Where do critical weights appear?")
    print("="*70)

    # Count by module type
    module_types = {}
    for r in results:
        # Extract module type (e.g., 'attn', 'mlp', 'gate', 'embed')
        parts = r.module.lower()
        if 'gate' in parts or 'router' in parts:
            mtype = 'MoE Router/Gate'
        elif 'attn' in parts or 'attention' in parts:
            if 'q_proj' in parts or 'k_proj' in parts:
                mtype = 'Attention Q/K'
            elif 'v_proj' in parts or 'o_proj' in parts:
                mtype = 'Attention V/O'
            else:
                mtype = 'Attention (other)'
        elif 'mlp' in parts or 'ffn' in parts:
            mtype = 'MLP/FFN'
        elif 'embed' in parts:
            mtype = 'Embedding'
        elif 'lm_head' in parts:
            mtype = 'LM Head'
        else:
            mtype = 'Other'

        module_types[mtype] = module_types.get(mtype, 0) + 1

    print("\n  Critical weights by module type:")
    for mtype, count in sorted(module_types.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 5)
        print(f"    {mtype:20s} {count:3d} ({pct:5.1f}%) {bar}")

    # Layer distribution
    layers = []
    for r in results:
        # Try to extract layer number
        parts = r.module.split('.')
        for p in parts:
            if p.isdigit():
                layers.append(int(p))
                break

    if layers:
        print(f"\n  Layer distribution:")
        print(f"    Min layer:  {min(layers)}")
        print(f"    Max layer:  {max(layers)}")
        print(f"    Avg layer:  {sum(layers)/len(layers):.1f}")

    print("\n" + "="*70 + "\n")

    # Save if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
