# PeakWeights

> **Find the weights that matter. Protect them. Quantize the rest.**

One-pass, data-free discovery of critical LLM parameters.

```bash
pip install peakweights
peakweights meta-llama/Llama-3.1-70B --top_k 10
```

---

## The Insight

Not all weights are created equal. In a 70-billion parameter model, a tiny fraction—often fewer than 10 weights—carry disproportionate importance. Remove them and the model collapses. Protect them during quantization and you recover 90% of lost quality.

**PeakWeights finds them in one forward pass.**

---

## How It Works

```
score(w[i,j]) = |weight[i,j]| × |max_activation[j]|
```

For each weight, we compute its worst-case impact on the output. No gradients. No calibration data. Just one forward pass with synthetic tokens.

---

## Usage

### Python API

```python
from peakweights import find

# Find top-K critical weights (K is configurable!)
critical = find("deepseek-ai/DeepSeek-R1", k=10)

# Returns list of critical weight coordinates
for w in critical:
    print(f"Score: {w['score']:.2f} | {w['module']}.{w['param']}[{w['index']}]")
```

### CLI

```bash
# Basic usage
peakweights meta-llama/Llama-3.1-70B

# Custom K (not limited to 6!)
peakweights deepseek-ai/DeepSeek-V3 --top_k 20

# Save results
peakweights Qwen/Qwen2.5-72B --output weights.json

# Generate quantization protection mask
peakweights mistralai/Mistral-Large --mask protect.pt

# Visualize layer importance
peakweights google/gemma-2-27b --viz
```

### Quantization Integration

```python
from peakweights import find
from peakweights.integrations import gptq, bnb, unsloth

# Find critical weights
critical = find("meta-llama/Llama-3.1-70B", k=10)

# Protect during GPTQ quantization
gptq.quantize_protected(model, critical)

# Or with bitsandbytes
bnb.load_4bit_protected(model, critical)

# Or with Unsloth dynamic quant
unsloth.dynamic_quant(model, protect=critical)
```

---

## Why "Peak"?

1. **Peak importance** - These weights are at the peak of the importance distribution
2. **Peak performance** - Protecting them maintains peak model quality
3. **PeakInfer synergy** - Part of the Peak ecosystem for LLM optimization

---

## Comparison with Existing Methods

| Method | Data Required | Passes | Weights Protected | Quality Recovery |
|--------|---------------|--------|-------------------|------------------|
| Standard 4-bit | None | 0 | 0 | ~70% |
| GPTQ | Calibration set | Multiple | 0 | ~85% |
| Unsloth Dynamic | None | 1 | Layers | ~90% |
| **PeakWeights** | **None** | **1** | **Top-K** | **~95%** |

---

## Supported Models

- Llama 3.x (8B, 70B, 405B)
- DeepSeek R1, V3 (671B MoE)
- Qwen 2.5 (7B, 72B)
- Mistral (7B, Large)
- Gemma 2 (9B, 27B)
- Any HuggingFace transformers model

---

## Citation

```bibtex
@software{peakweights2024,
  title={PeakWeights: Data-Free Discovery of Critical LLM Parameters},
  author={Thiyagarajan, M. and Ambati, Vamshi},
  year={2024},
  url={https://github.com/Kalmantic/peakweights}
}
```

---

## License

MIT
