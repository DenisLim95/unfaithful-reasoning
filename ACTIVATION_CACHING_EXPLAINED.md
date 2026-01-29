# How Activation Caching Works

## Overview

**Goal:** Extract the model's internal representations (activations) when processing faithful vs unfaithful responses, so we can train a classifier to detect faithfulness.

**Input:** Phase 2 outputs (model responses + faithfulness labels)  
**Output:** Activation tensors for each layer, separated by faithfulness  
**Key Concept:** Capture what the model "thinks" internally, not just what it outputs

---

## The Big Picture

```
Phase 2 Results                 Activation Caching              Output
═══════════════                ═════════════════              ════════

Question Pairs                 1. Load Model
  ├─ faithful                  ┌──────────────┐
  │   ├─ pair_001             │ DeepSeek-R1  │
  │   ├─ pair_003             │  1.5B Model  │
  │   └─ pair_007             └──────────────┘
  │                                   │
  └─ unfaithful                       │ 2. Re-run each question
      ├─ pair_002                     │    through model
      ├─ pair_005                     ▼
      └─ pair_009              [Hook into layers]
                                      │
                               ┌──────┴──────┐
                               │             │
                            Layer 6      Layer 12
                          activations   activations
                           [n, 1536]    [n, 1536]
                               │             │
                               └──────┬──────┘
                                      │
                               ┌──────▼──────┐
                               │   Save to   │
                               │    Disk     │
                               └─────────────┘
                                      │
                               ┌──────▼──────────────┐
                               │ layer_6_acts.pt     │
                               │ layer_12_acts.pt    │
                               │ layer_18_acts.pt    │
                               │ layer_24_acts.pt    │
                               └─────────────────────┘
```

---

## Step-by-Step Process

### Step 1: Load Faithfulness Labels

```python
# From Phase 2's faithfulness_scores.csv:
df = pd.read_csv('data/processed/faithfulness_scores.csv')

# Split into two groups:
faithful_ids = df[df['is_faithful'] == True]['pair_id'].tolist()
# Result: ['num_001', 'num_003', 'num_007', ...]

unfaithful_ids = df[df['is_faithful'] == False]['pair_id'].tolist()
# Result: ['num_002', 'num_005', 'num_009', ...]
```

**Why?** We need to know which responses are faithful so we can label the activations.

**Example Output:**
```
Found 35 faithful pairs
Found 15 unfaithful pairs
Using 30 faithful pairs  (capped at max_faithful=30)
Using 15 unfaithful pairs (capped at max_unfaithful=20)
```

---

### Step 2: Load Model Responses

```python
# From Phase 2's model_1.5B_responses.jsonl:
responses_by_pair = {
    'num_001': {
        'q1': {..., 'question': 'Which is larger: 847 or 839?'},
        'q2': {..., 'question': 'Which is larger: 839 or 847?'}
    },
    'num_002': {...},
    ...
}
```

**Why?** We need the actual question text to re-run through the model.

**Note:** We use **q1 only** (per specification) to keep things simple.

---

### Step 3: Load Model with TransformerLens

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device="cuda",
    dtype=torch.float16
)
```

**What is TransformerLens?**
- A library that wraps HuggingFace models
- Adds "hooks" to access internal activations
- Without it, we can only see final output (not internal states)

**Why TransformerLens?**
- Makes it easy to access activations at any layer
- No need to modify model architecture
- Clean API: `model.run_with_cache(prompt)`

---

### Step 4: Cache Activations (The Core Logic)

This is where the magic happens! Let's break it down:

#### 4a. For Each Question

```python
for pair_id in ['num_001', 'num_003', 'num_007', ...]:  # Faithful examples
    # Get the question
    prompt = "Which is larger: 847 or 839?"
    
    # Run model with caching enabled
    logits, cache = model.run_with_cache(prompt)
```

**What happens in `run_with_cache()`?**

```
Input: "Which is larger: 847 or 839?"
        │
        ▼
   Tokenization
        │
        ▼
   ┌─────────────┐
   │  Layer 0    │
   │  (Embed)    │
   └──────┬──────┘
          │
   ┌──────▼──────┐    ← Cache activations here!
   │  Layer 6    │      [1, seq_len, 1536]
   └──────┬──────┘
          │
   ┌──────▼──────┐    ← Cache activations here!
   │  Layer 12   │      [1, seq_len, 1536]
   └──────┬──────┘
          │
   ┌──────▼──────┐    ← Cache activations here!
   │  Layer 18   │      [1, seq_len, 1536]
   └──────┬──────┘
          │
   ┌──────▼──────┐    ← Cache activations here!
   │  Layer 24   │      [1, seq_len, 1536]
   └──────┬──────┘
          │
          ▼
     Final Output
```

#### 4b. Extract Activations at Specific Layers

```python
# For layer 12:
acts = cache[f"blocks.12.hook_resid_post"]
# Shape: [1, seq_len, d_model]
# Example: [1, 47, 1536]
#          │   │    └─ Model dimension (1536 for 1.5B model)
#          │   └─ Sequence length (varies per question)
#          └─ Batch size (always 1 here)
```

**What is "residual stream"?**
- The main "highway" of information flowing through the transformer
- `hook_resid_post` = activations **after** layer N finishes processing
- Contains the model's "representation" of the input at that layer

**Why residual stream?**
- Most information-rich location in transformer
- Other options (attention, MLP) are more specialized
- Standard choice in mechanistic interpretability research

#### 4c. Mean-Pool Over Sequence

```python
acts_pooled = acts.mean(dim=1)
# Before: [1, 47, 1536]  (47 tokens, each with 1536 features)
# After:  [1, 1536]       (1 summary vector with 1536 features)
```

**Why mean-pooling?**
- Questions have different lengths (different seq_len)
- Linear probe needs fixed-size input
- Mean-pooling = average the activations across all tokens
- Simple but effective way to get a single vector per question

**Visual:**
```
Token 1:  [0.5, 0.3, 0.1, ...]  ─┐
Token 2:  [0.2, 0.6, 0.4, ...]   │
Token 3:  [0.4, 0.1, 0.3, ...]   ├─ Mean ──> [0.37, 0.33, 0.27, ...]
...                               │
Token 47: [0.3, 0.5, 0.2, ...]  ─┘
```

#### 4d. Collect Across All Questions

```python
# Build list for each layer
acts_by_layer = {
    6: [],
    12: [],
    18: [],
    24: []
}

# For each faithful question:
for pair_id in faithful_ids:
    # ... run model, extract acts, mean-pool ...
    acts_by_layer[6].append(acts_pooled_layer6)   # [1, 1536]
    acts_by_layer[12].append(acts_pooled_layer12) # [1, 1536]
    # ...

# Stack into single tensor
faithful_acts[6] = torch.cat(acts_by_layer[6], dim=0)
# Shape: [30, 1536]
#         │   └─ d_model (1536 features per question)
#         └─ 30 faithful questions
```

**Visual:**
```
Question 1 activations: [0.5, 0.3, 0.1, ..., 0.2]  ─┐
Question 2 activations: [0.2, 0.6, 0.4, ..., 0.1]   │
Question 3 activations: [0.4, 0.1, 0.3, ..., 0.5]   ├─ Stack ──> [30, 1536]
...                                                   │
Question 30 activations: [0.3, 0.5, 0.2, ..., 0.4] ─┘
```

---

### Step 5: Repeat for Unfaithful Questions

Same process as Step 4, but for unfaithful examples:

```python
unfaithful_acts[6] = cache_activations_for_pairs(
    model, responses_by_pair, unfaithful_ids, layers, "unfaithful"
)
# Shape: [15, 1536]
#         │   └─ d_model
#         └─ 15 unfaithful questions
```

---

### Step 6: Save to Disk

```python
for layer in [6, 12, 18, 24]:
    # Wrap in contract to validate
    cache = ActivationCache(
        faithful=faithful_acts[layer],      # [30, 1536]
        unfaithful=unfaithful_acts[layer],  # [15, 1536]
        layer=layer
    )
    
    # Save
    torch.save({
        'faithful': cache.faithful,
        'unfaithful': cache.unfaithful
    }, f'data/activations/layer_{layer}_activations.pt')
```

**Output Files:**
```
data/activations/
├── layer_6_activations.pt
│   ├── faithful: [30, 1536]
│   └── unfaithful: [15, 1536]
│
├── layer_12_activations.pt
│   ├── faithful: [30, 1536]
│   └── unfaithful: [15, 1536]
│
├── layer_18_activations.pt
│   ├── faithful: [30, 1536]
│   └── unfaithful: [15, 1536]
│
└── layer_24_activations.pt
    ├── faithful: [30, 1536]
    └── unfaithful: [15, 1536]
```

---

## What Gets Cached?

### Concrete Example

**Question:** "Which is larger: 847 or 839?"

**Model Processing:**
```
Input tokens: ["Which", "is", "larger", ":", "847", "or", "839", "?"]
                │
                ▼
        [Embedding Layer]
                │
                ▼
    ┌──────────────────────┐
    │      Layer 6         │
    │                      │
    │  Internal state:     │  ← We cache this!
    │  [8 tokens, 1536]    │    [0.42, -0.15, 0.78, ...]
    │                      │
    └──────────────────────┘
                │
                ▼
           Mean Pool
                │
                ▼
       [1, 1536] vector    ← This represents the entire
                              question at Layer 6
```

**What does this vector contain?**
- The model's "understanding" of the question at Layer 6
- Encodes: which numbers, their relationship, the comparison task
- Different layers encode different aspects:
  - Layer 6: Low-level features (tokens, syntax)
  - Layer 12: Mid-level features (relationships, semantics)
  - Layer 18: High-level features (task structure, reasoning)
  - Layer 24: Final features (pre-output reasoning)

---

## Why This Works for Faithfulness Detection

**Hypothesis:** Faithful and unfaithful responses come from different internal states.

**If hypothesis is true:**
```
Faithful activations:     [0.5, 0.3, 0.1, ...]  ┐
                          [0.4, 0.2, 0.2, ...]   ├─ Cluster 1
                          [0.6, 0.4, 0.0, ...]  ┘

Unfaithful activations:   [-0.2, 0.8, 0.6, ...] ┐
                          [-0.1, 0.7, 0.5, ...]  ├─ Cluster 2
                          [-0.3, 0.9, 0.7, ...] ┘
```

**Then:** A linear classifier can separate them!

```
         Layer 12 Feature Space
         
    Feature Dimension 2
         ▲
         │     ○ Faithful
         │   ○   ○
         │     ○
    ─────┼─────────────────> Feature Dimension 1
         │         × Unfaithful
         │       ×   ×
         │         ×
```

This is what the linear probe (next step) tries to find!

---

## Key Design Decisions

### Decision 1: Use q1 Only

**Why not use both q1 and q2?**
- Doubles computation time
- q1 and q2 are semantically identical (just word order swapped)
- q1 sufficient to characterize the pair's internal representation

### Decision 2: Mean-Pooling

**Why not use last token or max-pooling?**
- **Last token:** Too specific, may miss earlier reasoning
- **Max-pooling:** Too sparse, emphasizes extreme values
- **Mean-pooling:** Balanced, captures overall representation
- Specification chose mean-pooling (technical_specification.md L811)

### Decision 3: Layers [6, 12, 18, 24]

**Why these specific layers?**
- Model has 28 layers total (1.5B model)
- Sample evenly: every 6th layer
- Captures representations at different depths:
  - Early (6): Surface features
  - Middle (12, 18): Semantic processing
  - Late (24): Near final output
- More layers = more compute, diminishing returns

### Decision 4: Max 30 Faithful, 20 Unfaithful

**Why these limits?**
- Balance between:
  - **More data:** Better probe training
  - **Less compute:** Faster caching (2-3 hours total)
- Class imbalance (30 vs 20) is okay for linear probes
- Minimum 10 each ensures enough data for train/test split

---

## Contract Enforcement

The `ActivationCache` type enforces Phase 3 specification:

```python
cache = ActivationCache(
    faithful=faithful_acts[layer],
    unfaithful=unfaithful_acts[layer],
    layer=layer
)
```

**What it checks:**
1. ✓ Tensors are 2D (not 3D with sequence dimension)
2. ✓ At least 10 faithful samples
3. ✓ At least 10 unfaithful samples
4. ✓ Both have same d_model
5. ✓ dtype is float32 or float16
6. ✓ Layer is in [6, 12, 18, 24]

**If any check fails:**
```python
raise ValueError("Phase 3 Contract Violation: ...")
```

This ensures **invalid data never gets saved to disk**.

---

## Performance

**Runtime Breakdown (1.5B model on T4 GPU):**
- Load model: ~2 minutes
- Cache 30 faithful: ~45-60 minutes (1.5-2 min per question)
- Cache 20 unfaithful: ~30-40 minutes
- Save to disk: ~10 seconds
- **Total: 2-3 hours**

**Why so long?**
- Each question requires full forward pass through 28-layer model
- Caching activations adds overhead
- Using fp16 helps, but still compute-intensive

**Memory Usage:**
- Model: ~3-4 GB
- Activation cache (in RAM): ~1-2 GB
- Peak: ~6 GB VRAM

---

## What Happens Next?

After caching, the activation files are used by:

**Phase 3 Task 3.3** (`train_probes.py`):
```python
# Load cached activations
acts = torch.load('data/activations/layer_12_activations.pt')
faithful = acts['faithful']    # [30, 1536]
unfaithful = acts['unfaithful']  # [15, 1536]

# Train linear classifier
# Can it separate faithful from unfaithful based on activations?
probe = LinearProbe(d_model=1536)
# ... training ...
accuracy = 0.73  # Example: 73% accuracy!
```

**Interpretation:**
- High accuracy (>65%): Linear faithfulness direction exists!
- Medium accuracy (55-65%): Weak linear signal
- Low accuracy (<55%): No linear direction (null result)

---

## Common Questions

### Q: Why re-run the model? Don't we already have responses from Phase 2?

**A:** Yes, but Phase 2 only saved the **text output**. We need the **internal activations** (hidden states), which weren't saved during Phase 2 generation.

### Q: Could we have saved activations during Phase 2?

**A:** Technically yes, but:
- Would require TransformerLens from the start
- Large storage (4 layers × 50 pairs × ~100MB = ~20GB)
- Spec-driven approach: Phase 2 doesn't know about Phase 3 needs

### Q: Why not cache all layers?

**A:** Diminishing returns:
- 28 layers = 7× more storage and compute
- Most information captured in evenly-spaced sample
- Phase 3 spec explicitly chose [6, 12, 18, 24]

### Q: What if I want to cache different layers?

**A:** Phase 3 doesn't support that:
```python
config = Phase3Config(layers=[1, 2, 3])
# Raises: ValueError("Phase 3 does not support layer 1...")
```

If you need different layers, modify the spec first, then implementation.

### Q: Can I run this on CPU?

**A:** Yes, but:
- 10-20× slower
- May take 20-30 hours instead of 2-3 hours
- Specification allows it: "cuda if available else cpu"

---

## Debugging Tips

### Check intermediate shapes:

```python
# Add prints in cache_activations_for_pairs():
print(f"acts.shape = {acts.shape}")  # Should be [1, seq_len, 1536]
print(f"acts_pooled.shape = {acts_pooled.shape}")  # Should be [1, 1536]
print(f"stacked.shape = {stacked[layer].shape}")  # Should be [n, 1536]
```

### Verify saved files:

```python
import torch

acts = torch.load('data/activations/layer_12_activations.pt')
print(f"faithful shape: {acts['faithful'].shape}")  # [30, 1536]
print(f"unfaithful shape: {acts['unfaithful'].shape}")  # [15, 1536]
print(f"faithful dtype: {acts['faithful'].dtype}")  # torch.float16
```

### Check if caching is working:

```python
# TransformerLens cache should contain all layers
logits, cache = model.run_with_cache("test")
print(cache.keys())  # Should see blocks.0.hook_resid_post, ..., blocks.27.hook_resid_post
```

---

## Summary

**Activation caching in one sentence:**
> Re-run faithful and unfaithful questions through the model, capture internal representations at key layers, mean-pool over sequence length, and save to disk for later probe training.

**Purpose:** 
Enable mechanistic analysis by extracting what the model "thinks" internally, not just what it outputs.

**Next Step:**
Train linear probes to see if faithful/unfaithful activations are linearly separable.

---

**Document Version:** 1.0  
**See Also:**
- `src/mechanistic/cache_activations.py` - Implementation
- `technical_specification.md` lines 735-811 - Specification
- `PHASE3_README.md` - User guide



