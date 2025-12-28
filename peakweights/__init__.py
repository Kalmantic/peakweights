"""
PeakWeights: Find the weights that matter. Protect them. Quantize the rest.

One-pass, data-free discovery of critical LLM parameters.

Usage:
    >>> from peakweights import find
    >>> critical = find("meta-llama/Llama-3.1-8B", k=10)
    >>> print(critical[0])
    {'rank': 1, 'score': 1247.3, 'module': 'layers.0.attn.q_proj', ...}

CLI:
    $ peakweights meta-llama/Llama-3.1-70B --top_k 10
    $ peakweights deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -k 10 -o results.json
"""

from .peakweights import (
    find,
    generate_protection_mask,
    visualize,
    PeakWeightsFinder,
    CriticalWeight,
    print_results,
    Colors,
    format_params,
    format_bytes,
    main,
)

__version__ = "0.3.0"
__all__ = [
    "find",
    "generate_protection_mask",
    "visualize",
    "PeakWeightsFinder",
    "CriticalWeight",
    "print_results",
    "Colors",
    "format_params",
    "format_bytes",
    "main",
]
