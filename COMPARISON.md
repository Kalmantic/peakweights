# PeakWeights vs Unsloth Dynamic Quantization

A detailed comparison of two complementary approaches to selective quantization protection.

---

## Overview

| Aspect | **PeakWeights** | **Unsloth Dynamic Quant** |
|--------|-----------------|---------------------------|
| **Granularity** | Individual weights | Entire layers |
| **Data Required** | None (data-free) | Calibration dataset (300K-1.5M tokens in v2.0) |
| **Analysis Method** | Forward pass importance scoring | Error plot analysis (activation + weight) |
| **Selection Criteria** | `\|weight\| × \|max_activation\|` | Quantization error spikes |
| **Output** | Top-K weight coordinates | List of layers to skip |
| **Typical Protection** | 20-100 weights (architecture dependent) | 10-50 layers |

---

## How Unsloth Dynamic Quantization Works

Based on analysis of Unsloth's blog posts and code:

### 1. Error Analysis Approach

Unsloth generates two plots per model:
- **Activation Quantization Errors**: Measures how much layer activations change after quantization
- **Weight Quantization Errors**: Measures per-layer weight quantization distortion

### 2. Key Observations from Unsloth's Analysis

**Qwen2 VL 2B:**
- Large activation errors in first few layers
- Gradual decrease in activation errors
- One parameter with large weight quantization error
- Result: Skip first few layers + vision projection

**Llama 3.2 11B Vision:**
- Vision encoder has one large spike
- Cross-attention output projections (all layers except first) should not be quantized

**Pixtral 12B:**
- Entire vision encoder should not be quantized to 4-bit
- Large activation errors across vision encoder

### 3. Dynamic v2.0 Improvements

From April 2025:
- Selectively quantizes **every layer** with different schemes
- Uses model-specific calibration (300K-1.5M tokens)
- Measures via 5-shot MMLU and KL Divergence
- Each model gets custom-tailored layer quantization

---

## How PeakWeights Works

### 1. Per-Weight Importance Scoring

```
score(w[i,j]) = |weight[i,j]| × |max_activation[j]|
```

This formula captures:
- **|weight|**: How large is this parameter?
- **|max_activation|**: How much can this weight affect output?

### 2. Single Forward Pass

1. Load model
2. Create synthetic input (16 tokens)
3. Register hooks on Linear/Embedding layers
4. Run one forward pass
5. Score all weights during hooks
6. Maintain top-K heap
7. Return critical weight coordinates

### 3. Data-Free Advantage

- No calibration dataset needed
- Works on any model immediately
- Results in seconds, not minutes

---

## Complementary Strengths

### What Unsloth Does Better

1. **Layer-Level Protection**: More practical for current quantization tools (GGUF, AWQ, GPTQ all work at layer granularity)

2. **Calibration-Based**: Better alignment with actual use patterns when calibration data matches deployment

3. **Battle-Tested**: 100M+ downloads, proven on Llama, DeepSeek, Qwen, Gemma

4. **Integration**: Direct GGUF export, works with Ollama/llama.cpp

### What PeakWeights Does Better

1. **Surgical Precision**: Protect only the 20-100 weights that matter most (architecture dependent), not entire layers

2. **Truly Data-Free**: No calibration dataset = faster iteration, no data dependency

3. **Minimal Memory Overhead**: Protecting 50 FP16 weights vs 10 full layers

4. **Research Insight**: Reveals which specific parameters carry disproportionate importance

---

## Combined Approach: The Best of Both

```python
# Step 1: Use PeakWeights for critical individual weights
from peakweights import find
critical = find("model", k=50)  # K depends on architecture

# Step 2: Use Unsloth Dynamic for layer-level decisions
from unsloth import FastLanguageModel
model = FastLanguageModel.from_pretrained("model", load_in_4bit=True)

# Step 3: Hybrid protection
# - Layers selected by Unsloth stay in higher precision
# - Within quantized layers, PeakWeights' top-K stay FP16
```

---

## Key Learnings from Unsloth's Approach

### 1. Vision Models Are Special

Unsloth found that vision encoders are particularly sensitive:
- Pixtral: Entire vision encoder should stay high precision
- Llama Vision: Cross-attention projections need protection
- Qwen VL: First few layers are critical

**PeakWeights Implication**: May need special handling for multimodal models.

### 2. Error Spikes Are Predictive

Single large spikes in error plots correlate with model breakage.

**PeakWeights Connection**: Our `|weight| × |max_activation|` naturally catches these—weights with extreme activations get high scores.

### 3. Calibration Dataset Matters for GGUF

Unsloth v2.0 uses 300K-1.5M tokens of curated data per model. This matters for:
- imatrix generation
- Layer-level importance estimation
- Chat-specific optimization

**PeakWeights Advantage**: Our synthetic input (16 tokens) is sufficient for weight-level importance.

### 4. Model-Specific Tuning

Each model family has different optimal quantization patterns:
- Gemma 3: Different layers than Llama 4
- MoE models: Router weights are disproportionately important
- Vision models: Cross-modal projections need protection

**PeakWeights Finding**: We expect critical weights to cluster in:
- Router/gate weights (MoE)
- Early attention layers
- Output projections

---

## Experimental Results

We tested PeakWeights on 4 architectures with K=50:

| Model | FP16 PPL | 4-bit PPL | PeakWeights | Recovery |
|-------|----------|-----------|-------------|----------|
| SmolLM2-1.7B | 13.29 | 16.31 | 13.32 | **99%** |
| Qwen2.5-7B | 9.41 | 10.05 | 9.44 | **96%** |
| DeepSeek-R1-7B | 44.52 | 46.12 | 44.60 | **95%** |
| Mistral-7B | 9.63 | 9.80 | 9.70 | 61% |

### Key Finding: K is Architecture-Dependent

| Model | K for 90% Recovery |
|-------|-------------------|
| SmolLM2-1.7B | K=20 |
| Qwen2.5-7B | K=50 |
| DeepSeek-R1-7B | K=50 |
| Mistral-7B | K=100 |

### Critical Weight Locations

| Model | MLP | Attention | lm_head |
|-------|-----|-----------|---------|
| Qwen/DeepSeek | 100% | 0% | 0% |
| Mistral | 0% | 0% | 100% |
| SmolLM2 | 66% | 34% | 0% |

Architecture determines where critical weights cluster.

---

## Summary

| Approach | Best For | Use When |
|----------|----------|----------|
| **Unsloth Dynamic** | Production deployment | Need GGUF, layer-level tools, proven reliability |
| **PeakWeights** | Research, extreme compression | Need surgical precision, no calibration data, understanding model structure |
| **Combined** | Maximum quality | Can afford both layer + weight protection |

The two approaches are **complementary, not competitive**. Unsloth protects at coarse granularity (layers), PeakWeights at fine granularity (weights). Together they can achieve better quality than either alone.

---

## References

- [Unsloth Dynamic 4-bit (Dec 2024)](https://unsloth.ai/blog/dynamic-4bit)
- [Unsloth Dynamic v2.0 (Apr 2025)](https://unsloth.ai/blog/dynamic-v2)
- [The Super Weight Paper (arXiv 2411.07191)](https://arxiv.org/abs/2411.07191)
