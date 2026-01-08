#!/usr/bin/env python3
"""
Convert experimental results to .pwi files for web download.
"""

import json
from datetime import datetime
from pathlib import Path

# Read the experimental results
with open("paper/peakweight run2 /peakweights_final_results.json", "r") as f:
    data = json.load(f)

output_dir = Path("data/analyzed")
output_dir.mkdir(parents=True, exist_ok=True)

# Model parameter counts
PARAM_COUNTS = {
    "Qwen/Qwen2.5-7B": 7_000_000_000,
    "mistralai/Mistral-7B-v0.3": 7_000_000_000,
    "HuggingFaceTB/SmolLM2-1.7B": 1_700_000_000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 7_000_000_000,
    "microsoft/Phi-3-mini-4k-instruct": 3_800_000_000,
}

# Filename mapping
FILENAMES = {
    "Qwen/Qwen2.5-7B": "qwen2.5-7b",
    "mistralai/Mistral-7B-v0.3": "mistral-7b",
    "HuggingFaceTB/SmolLM2-1.7B": "smollm2-1.7b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1-7b",
    "microsoft/Phi-3-mini-4k-instruct": "phi-3-mini",
}

for result in data["results"]:
    model_id = result["model_id"]

    if "error" in result:
        print(f"Skipping {model_id} - has error")
        continue

    # Build PWI structure
    weights = result.get("critical_weights", [])[:50]  # Top 50

    # Compute summary
    by_type = {"mlp": 0, "attention": 0, "embed": 0, "output": 0, "other": 0}
    layers = []
    score_total = 0

    for w in weights:
        module = w.get("module", "").lower()
        score_total += w.get("score", 0)

        # Categorize
        if "mlp" in module or "ffn" in module:
            by_type["mlp"] += 1
        elif "attn" in module or "attention" in module:
            by_type["attention"] += 1
        elif "embed" in module:
            by_type["embed"] += 1
        elif "lm_head" in module or "output" in module:
            by_type["output"] += 1
        else:
            by_type["other"] += 1

        # Extract layer number
        for part in w.get("module", "").split("."):
            if part.isdigit():
                layers.append(int(part))
                break

    # Remove zero counts
    by_type = {k: v for k, v in by_type.items() if v > 0}

    # Format total params
    total_params = PARAM_COUNTS.get(model_id, 0)
    if total_params >= 1e9:
        params_str = f"{total_params/1e9:.1f}B"
    elif total_params >= 1e6:
        params_str = f"{total_params/1e6:.1f}M"
    else:
        params_str = str(total_params)

    pwi_data = {
        "peakweights": "1.0",
        "model": model_id,
        "k": len(weights),
        "generated": datetime.now().isoformat(),
        "total_params": params_str,
        "weights": [
            {
                "rank": w.get("rank", i + 1),
                "score": round(w.get("score", 0), 2),
                "module": w.get("module", ""),
                "position": w.get("index", [0, 0]),
                "value": 0.0,  # Not stored in raw results
            }
            for i, w in enumerate(weights)
        ],
        "summary": {
            "by_type": by_type,
            "score_total": round(score_total, 2),
        },
        "metrics": {
            "fp16_ppl": round(result.get("fp16_ppl", 0), 2),
            "int4_ppl": round(result.get("int4_ppl", 0), 2),
            "recovery_at_50": round(result.get("recovery_rates", {}).get("50", {}).get("recovery_rate", 0) * 100, 1),
            "power_law_exponent": round(result.get("power_law", {}).get("exponent", 0), 3),
        },
    }

    if layers:
        pwi_data["summary"]["layer_range"] = [min(layers), max(layers)]

    # Write to file
    filename = FILENAMES.get(model_id, model_id.split("/")[-1].lower())
    output_path = output_dir / f"{filename}.pwi"

    with open(output_path, "w") as f:
        json.dump(pwi_data, f, indent=2)

    print(f"Created {output_path}")

print(f"\nCreated {len(list(output_dir.glob('*.pwi')))} .pwi files in {output_dir}")
