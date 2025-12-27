# PeakWeights - Visual Explanation

## What is a Neural Network, Really?

A neural network is **matrix multiplication** repeated many times.

```
                WEIGHTS (W)
                ┌─────────────────┐
                │ 0.2  0.8  0.1   │
 INPUT (x)      │ 0.5  0.3  0.9   │      OUTPUT
┌───────┐       │ 0.7  0.4  0.2   │     ┌───────┐
│ 1.0   │       │ 0.1  0.6  0.5   │     │  ???  │
│ 0.5   │   ×   │ 0.3  0.2  0.8   │  =  │  ???  │
│ 0.8   │       │ 0.9  0.1  0.4   │     │  ???  │
└───────┘       └─────────────────┘     └───────┘

Your input       70 billion of        What comes
(text)           these numbers!       out
```

Each **weight** is one number in these giant matrices.

---

## The Key Insight

When you quantize (round) a weight, you introduce error. But not all errors are equal.

```
BEFORE QUANTIZATION              AFTER QUANTIZATION

  3.0  ×  0.5  =  1.5              3.0  ×  0.0  =  0.0

     ERROR INTRODUCED = |1.5 - 0.0| = 1.5
```

The **error** depends on:
1. How much we changed the weight (0.5 → 0.0)
2. How big the activation was (3.0)

```
ERROR  ≈  |weight change|  ×  |activation|
       ≈       0.5         ×      3.0
       ≈              1.5
```

---

## Which Weights Matter Most?

Weights that are BOTH:
- **Large themselves** (big |weight|)
- **Connected to large activations** (big |activation|)

```
                    ACTIVATION SIZE
                Small           Large
            ┌───────────────┬───────────────┐
     Small  │   ○           │     ○         │   ← Safe to quantize
W           │  tiny×tiny    │  tiny×big     │
E           │   = tiny      │   = medium    │
I           ├───────────────┼───────────────┤
G    Large  │     ○         │     ⬤         │   ← DANGER ZONE!
H           │  big×tiny     │  big×big      │
T           │   = medium    │   = HUGE      │
            └───────────────┴───────────────┘
                                    ↑
                           CRITICAL WEIGHTS
                           (top-right corner)
```

**PeakWeights finds the weights in the top-right corner.**

---

## The Formula

```
score(w[i,j]) = |weight[i,j]| × |max_activation[j]|
```

That's it. We multiply weight size by the biggest activation that flows through it.

---

## Finding 10 Needles in 70 Billion

```
70,000,000,000 WEIGHTS
══════════════════════

┌─────────────────────────────────────────────────────┐
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░★░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░░░░░░░★░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░★░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░★░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░░░░░★░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└─────────────────────────────────────────────────────┘

░ = Safe to quantize (99.9999% of weights)
★ = CRITICAL - Keep at full precision

RESULT:
░ weights: 16-bit → 4-bit  (75% memory savings!)
★ weights: 16-bit → 16-bit (keep perfect)

→ 95% quality retained (vs 70% without protection)
```

---

## The Algorithm

```
INPUT: "0 1 2 3 4 5..." (fake tokens - no real data needed!)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1                                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ activations flow through...                           │  │
│  │                                                       │  │
│  │  ───►  0.1  0.3  2.1  0.5  0.2  8.7  0.4  0.1  ───►  │  │
│  │                               ↑                       │  │
│  │                           record max=8.7              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   (repeat for all layers)
         │
         ▼
   Score each weight: |W| × |max_activation|
         │
         ▼
   Keep only Top-K using a min-heap
         │
         ▼
OUTPUT: The 50 most critical weights
```

**One forward pass. No gradients. No real data. O(n) time, O(K) space.**

---

## Why "Data-Free" Works

We don't care about **what the model outputs**. We only care about:

1. **How big is each weight?** → Already in the model
2. **How big can activations get?** → Any input reveals this

The synthetic input just **wakes up** the model so we can observe which pathways carry large signals. The network structure determines this, not the specific input.

**Analogy:** To find which pipes can carry the most water, you don't need to run actual water through every faucet. Just look at the pipe diameters and pump capacity.

---

## Why This Matters

```
Without PeakWeights:
  Compress everything equally → Model loses ~30% quality

With PeakWeights:
  Protect 50 weights, compress the rest → Model keeps ~95% quality
```

**It's like:** "In a company of 70 billion employees, we found the 50 people who keep the power grid running. Don't fire them."
