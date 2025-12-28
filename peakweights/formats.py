"""
PeakWeights Format Handlers

The `.pwi` format: JSON that's web-ready, grep-able, and PyTorch-friendly.

Usage:
    # Python
    from peakweights import find, save, load, get_skip_modules

    weights = find("Qwen/Qwen2.5-7B", k=50)
    save(weights, "critical.pwi", model_name="Qwen/Qwen2.5-7B")

    loaded = load("critical.pwi")
    skip = get_skip_modules(loaded, top_n=5)

    # CLI
    peakweights Qwen/Qwen2.5-7B -o weights.pwi
    peakweights Qwen/Qwen2.5-7B -o weights.pt
    peakweights Qwen/Qwen2.5-7B -o weights.csv
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


FORMAT_VERSION = "1.0"


def save(
    weights: List[Dict[str, Any]],
    path: str,
    model_name: Optional[str] = None,
    total_params: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save critical weights to file. Format auto-detected from extension.

    Args:
        weights: List of weight dicts from peakweights.find()
        path: Output path (.pwi, .json, .pt, .csv)
        model_name: Model identifier for metadata
        total_params: Total model parameters
        metadata: Additional metadata

    Examples:
        >>> save(weights, "critical.pwi", model_name="Qwen/Qwen2.5-7B")
        >>> save(weights, "mask.pt")
        >>> save(weights, "weights.csv")
    """
    fmt = _detect_format(path)

    if fmt == "pwi":
        _save_pwi(weights, path, model_name, total_params, metadata)
    elif fmt == "pt":
        _save_pt(weights, path, model_name, metadata)
    elif fmt == "csv":
        _save_csv(weights, path)
    else:
        raise ValueError(f"Unknown format: {fmt}. Use .pwi, .json, .pt, or .csv")


def load(path: str) -> List[Dict[str, Any]]:
    """
    Load critical weights from file.

    Args:
        path: Input file path

    Returns:
        List of weight dictionaries

    Examples:
        >>> weights = load("critical.pwi")
        >>> weights = load("mask.pt")
    """
    fmt = _detect_format(path)

    if fmt == "pwi":
        return _load_pwi(path)
    elif fmt == "pt":
        return _load_pt(path)
    elif fmt == "csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def to_mask(weights: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Convert weights to protection mask for quantization.

    Returns:
        Dict mapping "module" -> [flat_indices]

    Example:
        >>> mask = to_mask(weights)
        >>> # Use with bitsandbytes, etc.
    """
    mask = {}
    for w in weights:
        module = w.get("module", "unknown")
        if module not in mask:
            mask[module] = []
        mask[module].append(w.get("flat_index", 0))
    return mask


def get_skip_modules(weights: List[Dict[str, Any]], top_n: int = 10) -> List[str]:
    """
    Get module names to skip during quantization.

    Args:
        weights: Critical weights list
        top_n: Number of unique modules to return

    Returns:
        List of module names for llm_int8_skip_modules

    Example:
        >>> skip = get_skip_modules(weights, top_n=5)
        >>> config = BitsAndBytesConfig(llm_int8_skip_modules=skip)
    """
    seen = set()
    modules = []
    for w in weights:
        module = w.get("module", "")
        if module and module not in seen:
            seen.add(module)
            modules.append(module)
            if len(modules) >= top_n:
                break
    return modules


def to_json(
    weights: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    total_params: Optional[int] = None,
) -> str:
    """
    Convert weights to JSON string (for web APIs).

    Example:
        >>> json_str = to_json(weights, model_name="Qwen/Qwen2.5-7B")
        >>> return Response(json_str, content_type="application/json")
    """
    data = _build_pwi(weights, model_name, total_params, None)
    return json.dumps(data, indent=2)


# =============================================================================
# Format Detection
# =============================================================================

def _detect_format(path: str) -> str:
    """Detect format from extension."""
    ext = Path(path).suffix.lower()
    return {
        ".pwi": "pwi",
        ".json": "pwi",  # JSON uses same format as PWI
        ".pt": "pt",
        ".pth": "pt",
        ".csv": "csv",
    }.get(ext, "pwi")


# =============================================================================
# PWI Format (PeakWeights Index) - Primary format
# =============================================================================

def _build_pwi(
    weights: List[Dict],
    model_name: Optional[str],
    total_params: Optional[int],
    metadata: Optional[Dict],
) -> Dict[str, Any]:
    """Build PWI data structure."""

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
    if total_params:
        if total_params >= 1e9:
            params_str = f"{total_params/1e9:.1f}B"
        elif total_params >= 1e6:
            params_str = f"{total_params/1e6:.1f}M"
        else:
            params_str = str(total_params)
    else:
        params_str = None

    data = {
        "peakweights": FORMAT_VERSION,
        "model": model_name or "unknown",
        "k": len(weights),
        "generated": datetime.now().isoformat(),
        "weights": [
            {
                "rank": w.get("rank", i + 1),
                "score": round(w.get("score", 0), 2),
                "module": w.get("module", ""),
                "position": w.get("index", [w.get("row", 0), w.get("col", 0)]),
                "value": round(w.get("value", 0), 6),
            }
            for i, w in enumerate(weights)
        ],
        "summary": {
            "by_type": by_type,
            "score_total": round(score_total, 2),
        },
    }

    if params_str:
        data["total_params"] = params_str

    if layers:
        data["summary"]["layer_range"] = [min(layers), max(layers)]

    if metadata:
        data["metadata"] = metadata

    return data


def _save_pwi(
    weights: List[Dict],
    path: str,
    model_name: Optional[str],
    total_params: Optional[int],
    metadata: Optional[Dict],
) -> None:
    """Save as PWI (JSON with schema)."""
    data = _build_pwi(weights, model_name, total_params, metadata)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_pwi(path: str) -> List[Dict]:
    """Load from PWI/JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, list):
        return data  # Old format: plain list

    weights = data.get("weights", [])

    # Normalize to internal format
    return [
        {
            "rank": w.get("rank", i + 1),
            "score": w.get("score", 0),
            "module": w.get("module", ""),
            "param": "weight",
            "index": w.get("position", [0, 0]),
            "row": w.get("position", [0, 0])[0] if w.get("position") else 0,
            "col": w.get("position", [0, 0])[1] if w.get("position") else 0,
            "value": w.get("value", 0),
            "flat_index": w.get("position", [0, 0])[0] * 10000 + w.get("position", [0, 0])[1] if w.get("position") else 0,
        }
        for i, w in enumerate(weights)
    ]


# =============================================================================
# PyTorch Format - For direct quantization use
# =============================================================================

def _save_pt(
    weights: List[Dict],
    path: str,
    model_name: Optional[str],
    metadata: Optional[Dict],
) -> None:
    """Save as PyTorch file with mask."""
    mask = to_mask(weights)

    data = {
        "version": FORMAT_VERSION,
        "model": model_name or "unknown",
        "k": len(weights),
        "mask": mask,
        "weights": weights,
        "skip_modules": get_skip_modules(weights, top_n=20),
    }

    if metadata:
        data["metadata"] = metadata

    torch.save(data, path)


def _load_pt(path: str) -> List[Dict]:
    """Load from PyTorch file."""
    data = torch.load(path, weights_only=False)

    if "weights" in data:
        return data["weights"]

    # Old mask-only format: reconstruct weights
    weights = []
    rank = 1
    for module, indices in data.items():
        if module in ("version", "model", "k", "mask", "metadata", "skip_modules"):
            continue
        for idx in indices:
            weights.append({
                "rank": rank,
                "module": module,
                "param": "weight",
                "flat_index": idx,
                "score": 0,
                "index": [0, 0],
                "value": 0,
            })
            rank += 1
    return weights


def load_mask(path: str) -> Dict[str, List[int]]:
    """
    Load just the protection mask from .pt file.

    Quick way to get skip modules for quantization.

    Example:
        >>> mask = load_mask("protect.pt")
        >>> skip = list(mask.keys())[:5]
    """
    data = torch.load(path, weights_only=False)
    return data.get("mask", to_mask(data.get("weights", [])))


# =============================================================================
# CSV Format - Spreadsheet export
# =============================================================================

def _save_csv(weights: List[Dict], path: str) -> None:
    """Save as CSV."""
    with open(path, 'w') as f:
        f.write("rank,score,module,row,col,value\n")
        for w in weights:
            idx = w.get("index", [w.get("row", 0), w.get("col", 0)])
            row = idx[0] if isinstance(idx, list) else w.get("row", 0)
            col = idx[1] if isinstance(idx, list) else w.get("col", 0)
            f.write(f"{w.get('rank', 0)},{w.get('score', 0):.4f},{w.get('module', '')},{row},{col},{w.get('value', 0):.6f}\n")


def _load_csv(path: str) -> List[Dict]:
    """Load from CSV."""
    weights = []
    with open(path, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            try:
                weights.append({
                    "rank": int(parts[0]),
                    "score": float(parts[1]),
                    "module": parts[2],
                    "param": "weight",
                    "row": int(parts[3]),
                    "col": int(parts[4]),
                    "index": [int(parts[3]), int(parts[4])],
                    "value": float(parts[5]),
                    "flat_index": int(parts[3]) * 10000 + int(parts[4]),
                })
            except (ValueError, IndexError):
                continue
    return weights
