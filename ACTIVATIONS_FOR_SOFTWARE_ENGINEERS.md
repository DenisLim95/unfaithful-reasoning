# Neural Network Activations Explained for Software Engineers

## Table of Contents
1. [What are Activations?](#what-are-activations)
2. [Why Layers 6, 12, 18, 24?](#why-layers-6-12-18-24)
3. [What is the Residual Stream?](#what-is-the-residual-stream)
4. [What is Mean Pooling?](#what-is-mean-pooling)

---

## What are Activations?

### Software Engineering Analogy

Think of a neural network like a **data processing pipeline**:

```python
# Traditional software pipeline
input_text = "Which is larger: 847 or 839?"

def process(text):
    tokens = tokenize(text)           # ["Which", "is", "larger", ...]
    features = extract_features(tokens)  # [0.5, 0.3, 0.1, ...]
    result = classify(features)       # "847"
    return result
```

**Activations are the intermediate values** at each step:

```python
# Neural network pipeline (simplified)
input_text = "Which is larger: 847 or 839?"

layer_0_output = embedding_layer(input_text)      # ← Activation at layer 0
layer_1_output = transformer_layer_1(layer_0_output)  # ← Activation at layer 1
layer_2_output = transformer_layer_2(layer_1_output)  # ← Activation at layer 2
...
layer_28_output = transformer_layer_28(layer_27_output)
final_answer = output_head(layer_28_output)       # "847"
```

**Activations = the output of each layer** (the intermediate state of processing).

---

### Concrete Example: Real Activations

Let's say we run the question "Which is larger: 847 or 839?" through the model.

**Input:**
```
Text: "Which is larger: 847 or 839?"
```

**Layer 6 Activations (actual shape):**
```python
[
  # Token 1: "Which"
  [0.42, -0.15, 0.78, 0.22, -0.56, ... (1536 numbers total)],
  
  # Token 2: "is"
  [0.31, 0.05, -0.43, 0.67, 0.12, ... (1536 numbers total)],
  
  # Token 3: "larger"
  [-0.22, 0.58, 0.33, -0.41, 0.87, ... (1536 numbers total)],
  
  # Token 4: ":"
  [0.15, -0.33, 0.52, 0.08, -0.71, ... (1536 numbers total)],
  
  # Token 5: "847"
  [0.88, 0.22, -0.15, 0.44, 0.03, ... (1536 numbers total)],
  
  # Token 6: "or"
  [0.11, -0.42, 0.67, -0.23, 0.55, ... (1536 numbers total)],
  
  # Token 7: "839"
  [0.76, 0.18, -0.28, 0.51, -0.09, ... (1536 numbers total)],
  
  # Token 8: "?"
  [0.03, 0.61, -0.44, 0.19, 0.82, ... (1536 numbers total)]
]
```

**Shape:** `[8, 1536]`
- 8 rows = 8 tokens in the input
- 1536 columns = 1536 "features" per token

Each number is a **float** between roughly -1 and 1.

---

### What Do These Numbers Mean?

Think of each of the 1536 numbers as a **feature detector**:

```python
# Hypothetical interpretation (simplified):
activation[0] = 0.42    # "How much this looks like a question word"
activation[1] = -0.15   # "How much this is a verb"
activation[2] = 0.78    # "How much this relates to comparison"
activation[3] = 0.22    # "How much this is about numbers"
...
activation[1535] = 0.15 # "Some other abstract feature"
```

**Real activations are much more abstract:**
- The model learns these features during training
- We don't know exactly what each number means
- But together, they represent the model's "understanding"

---

### Why Are Activations Useful?

**Analogy:** Debugging with print statements

```python
def process_order(order):
    validated = validate(order)
    print(f"After validation: {validated}")  # ← Check intermediate state
    
    enriched = enrich(validated)
    print(f"After enrichment: {enriched}")   # ← Check intermediate state
    
    result = finalize(enriched)
    print(f"Final result: {result}")         # ← Check final state
    
    return result
```

**Activations let us see what the model is "thinking" at each stage:**

```python
# Layer 6: "I see two numbers and a comparison question"
# Layer 12: "I understand this is asking which number is bigger"
# Layer 18: "I've determined 847 > 839"
# Layer 24: "I'm confident the answer is 847"
```

We can't read the exact thoughts, but we can:
1. Compare activations from faithful vs unfaithful responses
2. See if they're systematically different
3. Train a classifier to detect those differences

---

## Why Layers 6, 12, 18, 24?

### The Model Structure

The DeepSeek-R1-Distill-1.5B model has **28 layers** (0-27):

```
Layer 0  ─┐
Layer 1   │
Layer 2   │ Early layers
Layer 3   │
Layer 4   │
Layer 5   │
Layer 6  ─┘ ← We sample this (early processing)
Layer 7   ┐
Layer 8   │
Layer 9   │
Layer 10  │
Layer 11  │ Mid-early layers
Layer 12 ─┘ ← We sample this (mid-level understanding)
Layer 13  ┐
Layer 14  │
Layer 15  │
Layer 16  │
Layer 17  │ Mid-late layers
Layer 18 ─┘ ← We sample this (high-level reasoning)
Layer 19  ┐
Layer 20  │
Layer 21  │
Layer 22  │
Layer 23  │ Late layers
Layer 24 ─┘ ← We sample this (near-final reasoning)
Layer 25  ┐
Layer 26  │ Final layers
Layer 27 ─┘
```

### Why Not All Layers?

**Reason 1: Computational Cost**

Caching all 28 layers:
- **Storage:** ~20 GB (vs ~3 GB for 4 layers)
- **Time:** ~8-10 hours (vs ~2-3 hours for 4 layers)
- **Diminishing returns:** Neighboring layers are very similar

**Reason 2: Even Sampling**

We sample every **6th layer** to get even coverage:
- 6 = 28 / 4 (roughly)
- Captures different stages of processing

**Reason 3: Specification**

The original research (technical_specification.md) chose these layers based on:
- Prior mechanistic interpretability research
- Empirical results showing these layers are most informative
- Balance between coverage and efficiency

### What Each Layer Captures

Think of it like compilation stages:

```python
# Software compilation analogy
source_code = "Which is larger: 847 or 839?"

# Layer 6 (Early)
# Like: Lexical analysis / tokenization
# Captures: Individual words, basic syntax
tokens = ["Which", "is", "larger", ":", "847", "or", "839", "?"]

# Layer 12 (Mid-Early)
# Like: Parsing / AST construction
# Captures: Relationships between words, sentence structure
ast = ComparisonQuestion(
    type="which",
    operator="larger",
    operand1=847,
    operand2=839
)

# Layer 18 (Mid-Late)
# Like: Semantic analysis
# Captures: Meaning, intent, logic
semantics = {
    "task": "numerical_comparison",
    "operation": "max",
    "inputs": [847, 839],
    "expected_type": "integer"
}

# Layer 24 (Late)
# Like: Code generation / optimization
# Captures: Final reasoning, answer formation
reasoning = {
    "computation": "847 > 839",
    "result": 847,
    "confidence": "high"
}
```

### Why These Specific Numbers (6, 12, 18, 24)?

**Pattern:** `6 = 6×1, 12 = 6×2, 18 = 6×3, 24 = 6×4`

This creates even spacing:
```
0%    25%   50%   75%   100%
│     │     │     │     │
├─────┼─────┼─────┼─────┤
0     6    12    18    24   28 (layers)
      ↑     ↑     ↑     ↑
      Sample these
```

**Alternative we could have chosen (but didn't):**
- First/middle/last: [0, 14, 27]
- Denser sampling: [6, 9, 12, 15, 18, 21, 24]
- Random sampling: [5, 13, 19, 26]

The specification chose [6, 12, 18, 24] and Phase 3 implements exactly that (no more, no less).

---

## What is the Residual Stream?

### Software Engineering Analogy

Imagine a **Git repository** with a main branch:

```
main branch (always updated)
│
├─ commit 1: "Add tokenization"
│  └─ feature branch A: "Improve tokenizer"  ← Side branch (merge back)
│
├─ commit 2: "Add parsing"
│  └─ feature branch B: "Optimize parser"    ← Side branch (merge back)
│
├─ commit 3: "Add semantic analysis"
│  └─ feature branch C: "Better error handling" ← Side branch (merge back)
│
└─ commit 4: "Add code generation"
```

**Main branch = Residual stream** (the main flow of information)

**Feature branches = Specialized computations** (attention, feed-forward networks)

---

### In Transformer Architecture

```python
# Simplified transformer layer (pseudocode)
def transformer_layer(input_activations):
    # Main highway (residual stream)
    residual = input_activations
    
    # Side computation 1: Attention
    attention_output = self_attention(residual)
    residual = residual + attention_output  # Add back to main stream
    
    # Side computation 2: Feed-forward network
    ffn_output = feedforward_network(residual)
    residual = residual + ffn_output        # Add back to main stream
    
    return residual  # ← This is the residual stream
```

**Key insight:** Information **flows through** the residual stream, with each layer **adding to it**:

```
Input: [0.1, 0.2, 0.3, ...]

Layer 1: [0.1, 0.2, 0.3, ...] + [Δ from attention] + [Δ from FFN]
       = [0.15, 0.25, 0.35, ...]

Layer 2: [0.15, 0.25, 0.35, ...] + [Δ from attention] + [Δ from FFN]
       = [0.18, 0.28, 0.40, ...]

...and so on
```

---

### Why Cache the Residual Stream?

**Analogy:** Database queries

```python
# Poor approach: Query specialized tables
user_data = query_user_table(user_id)      # Only user info
order_data = query_order_table(user_id)    # Only order info
address_data = query_address_table(user_id) # Only address info

# Better approach: Query the main denormalized table
all_data = query_main_table(user_id)       # Everything in one place!
```

**Residual stream = the "main table" of the transformer:**
- Contains **all** information flowing through the network
- Attention and FFN are just specialized updates to it
- Most comprehensive view of the model's state

**Alternative caches we could use (but don't in Phase 3):**
- **Attention patterns:** Shows what tokens attend to each other
- **MLP activations:** Shows specialized feature transformations
- **Attention head outputs:** Shows what each head computed

We chose residual stream because:
1. Most information-rich
2. Standard in interpretability research
3. Simpler to work with (one stream vs. many components)

---

### Technical Detail: hook_resid_post

```python
# In TransformerLens, we cache this:
activations = cache[f"blocks.{layer}.hook_resid_post"]
```

**What does "post" mean?**

```python
def transformer_layer(input):
    # hook_resid_pre: BEFORE this layer's processing
    residual_pre = input
    
    # ... attention ...
    # ... feed-forward ...
    
    # hook_resid_post: AFTER this layer's processing
    residual_post = residual_pre + attention_out + ffn_out
    
    return residual_post
```

We use `hook_resid_post` because:
- Captures the **output** of this layer
- Includes all updates from attention and FFN
- Ready to be input to the next layer

---

## What is Mean Pooling?

### The Problem

Different questions have different numbers of tokens:

```python
Question 1: "Which is larger: 847 or 839?"
Tokens: ["Which", "is", "larger", ":", "847", "or", "839", "?"]
Length: 8 tokens

Question 2: "Compare the following two numbers and tell me which one is bigger: 847 or 839?"
Tokens: ["Compare", "the", "following", "two", "numbers", ..., "839", "?"]
Length: 16 tokens
```

**Activations have different shapes:**
```python
Question 1: [8, 1536]   # 8 tokens × 1536 features
Question 2: [16, 1536]  # 16 tokens × 1536 features
```

**Problem:** Linear probe needs **fixed-size input**!

```python
# This doesn't work:
probe = LinearProbe(input_size=???)  # What size???
```

---

### The Solution: Mean Pooling

**Mean pooling = Take the average** across all tokens.

### Concrete Example

**Before mean pooling:**
```python
activations = [
    [0.5, 0.3, 0.1],  # Token 1
    [0.2, 0.6, 0.4],  # Token 2
    [0.4, 0.1, 0.3],  # Token 3
    [0.3, 0.5, 0.2]   # Token 4
]
# Shape: [4, 3] (4 tokens, 3 features)
```

**After mean pooling:**
```python
pooled = [
    (0.5 + 0.2 + 0.4 + 0.3) / 4,  # Feature 1: 0.35
    (0.3 + 0.6 + 0.1 + 0.5) / 4,  # Feature 2: 0.375
    (0.1 + 0.4 + 0.3 + 0.2) / 4   # Feature 3: 0.25
]
# Result: [0.35, 0.375, 0.25]
# Shape: [3] (just 3 features, no token dimension!)
```

**Now all questions have the same shape:** `[1536]`

---

### Code Example

```python
import torch

# Before: Different lengths
q1_acts = torch.randn(8, 1536)   # 8 tokens
q2_acts = torch.randn(16, 1536)  # 16 tokens

# Mean pooling
q1_pooled = q1_acts.mean(dim=0)  # [1536]
q2_pooled = q2_acts.mean(dim=0)  # [1536]

# Now they're the same size!
print(q1_pooled.shape)  # torch.Size([1536])
print(q2_pooled.shape)  # torch.Size([1536])
```

---

### Why Mean (Average)?

**Alternative pooling methods:**

```python
# Max pooling: Take the maximum value
max_pooled = activations.max(dim=0)
# Result: [0.5, 0.6, 0.4]
# Pro: Captures strongest signal
# Con: Loses information about other tokens

# Min pooling: Take the minimum value
min_pooled = activations.min(dim=0)
# Result: [0.2, 0.1, 0.1]
# Pro: Captures weakest signal
# Con: Emphasizes noise

# Last token pooling: Just use the last token
last_pooled = activations[-1]
# Result: [0.3, 0.5, 0.2]
# Pro: Often contains summary information
# Con: Ignores all earlier tokens
```

**Mean pooling is balanced:**
- Incorporates **all** tokens
- Smooths out noise
- Standard in NLP (used in sentence embeddings)
- Works well empirically

---

### Visual Analogy

Think of it like **calculating a team's average score**:

```
Team scores over 4 games:
Game 1: 85 points
Game 2: 92 points
Game 3: 78 points
Game 4: 90 points

Average: (85 + 92 + 78 + 90) / 4 = 86.25

Instead of saying "the team scored 85, 92, 78, and 90",
we can summarize: "the team averages 86.25 points per game"
```

**Mean pooling does the same for tokens:**
```
Token 1 features: [0.5, 0.3, 0.1]
Token 2 features: [0.2, 0.6, 0.4]
Token 3 features: [0.4, 0.1, 0.3]
Token 4 features: [0.3, 0.5, 0.2]

Average: [0.35, 0.375, 0.25]

Instead of 4 separate token representations,
we get 1 summary representation.
```

---

### In Our Pipeline

```python
def cache_activations_for_pairs(...):
    for question in questions:
        # Run model
        logits, cache = model.run_with_cache(question)
        
        # Get activations for layer 12
        acts = cache["blocks.12.hook_resid_post"]
        # Shape: [1, num_tokens, 1536]
        # Example: [1, 8, 1536] for an 8-token question
        
        # Mean pool over tokens (dim=1)
        acts_pooled = acts.mean(dim=1)
        # Shape: [1, 1536]
        
        # Remove batch dimension
        acts_pooled = acts_pooled.squeeze(0)
        # Shape: [1536]
        
        # Store
        all_activations.append(acts_pooled)
```

**Result:**
```python
# Before mean pooling (can't stack these!)
q1: [8, 1536]
q2: [16, 1536]
q3: [12, 1536]

# After mean pooling (can stack!)
q1: [1536]
q2: [1536]
q3: [1536]

# Stack into dataset
dataset = torch.stack([q1, q2, q3])
# Shape: [3, 1536] ← Perfect for linear probe!
```

---

## Putting It All Together

### The Full Pipeline (Step-by-Step)

```python
# 1. Input question
question = "Which is larger: 847 or 839?"

# 2. Run through model with caching
logits, cache = model.run_with_cache(question)

# 3. Extract activations at layer 12
acts_layer12 = cache["blocks.12.hook_resid_post"]
# Shape: [1, 8, 1536]
#         │  │  └─ 1536 features (model dimension)
#         │  └─ 8 tokens in question
#         └─ batch size of 1

# 4. Mean pool over tokens
acts_pooled = acts_layer12.mean(dim=1)
# Shape: [1, 1536]
#         │  └─ 1536 features (averaged across all 8 tokens)
#         └─ batch size of 1

# 5. Remove batch dimension
acts_final = acts_pooled.squeeze(0)
# Shape: [1536]
# This single vector represents the entire question at layer 12!

# 6. Save to disk
torch.save({'faithful': faithful_acts, 'unfaithful': unfaithful_acts},
           'data/activations/layer_12_activations.pt')
```

---

## Summary for Software Engineers

| Concept | Software Engineering Analogy | Real Example |
|---------|------------------------------|--------------|
| **Activations** | Intermediate values in a pipeline | `[0.42, -0.15, 0.78, ...]` |
| **Layers** | Stages in compilation | Lexer → Parser → Analyzer → Generator |
| **Residual Stream** | Main Git branch (main flow) | `residual = input + attention + ffn` |
| **Mean Pooling** | Calculating average (summarizing) | `[8, 1536] → mean → [1536]` |
| **Why 6, 12, 18, 24?** | Even sampling across pipeline | 25%, 50%, 75%, 100% through 28 layers |

---

## Quick Reference

### Shapes Cheat Sheet

```python
# Input
question = "Which is larger: 847 or 839?"
tokens = ["Which", "is", "larger", ":", "847", "or", "839", "?"]
# 8 tokens

# After model forward pass (layer 12)
activations = cache["blocks.12.hook_resid_post"]
# Shape: [1, 8, 1536]
#         │  │  └─ d_model (1536 features per token)
#         │  └─ sequence length (8 tokens)
#         └─ batch size (always 1)

# After mean pooling
pooled = activations.mean(dim=1)
# Shape: [1, 1536]
#         │  └─ d_model (averaged across 8 tokens)
#         └─ batch size

# After squeeze
final = pooled.squeeze(0)
# Shape: [1536]
# Single vector representing the entire question!

# After collecting all questions (30 faithful)
faithful_dataset = torch.stack([q1, q2, q3, ..., q30])
# Shape: [30, 1536]
#         │   └─ d_model
#         └─ 30 questions
```

---

## Common Questions

### Q: Why not just use the final output?

**A:** The final output only tells us the answer. Activations tell us **how** the model arrived at that answer.

```python
# Final output
"847"  # This is the same for both faithful and unfaithful!

# Activations (layer 12)
Faithful:   [0.5, 0.3, 0.1, ...]  ← Different internal state
Unfaithful: [-0.2, 0.8, 0.6, ...] ← Different internal state
```

### Q: Can I visualize activations?

**A:** Sort of. 1536 dimensions is hard to visualize, but you can:

1. **Plot histograms:**
```python
import matplotlib.pyplot as plt
plt.hist(activations.flatten(), bins=50)
plt.title("Distribution of activation values")
```

2. **Use dimensionality reduction:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(activations)  # [1536] → [2]
plt.scatter(reduced[:, 0], reduced[:, 1])
```

3. **Look at statistics:**
```python
print(f"Mean: {activations.mean()}")
print(f"Std: {activations.std()}")
print(f"Min: {activations.min()}")
print(f"Max: {activations.max()}")
```

### Q: Are these activations always the same for the same input?

**A:** 
- With `temperature=0`: Yes, deterministic
- With `temperature>0`: No, slight randomness (but similar)
- In our case (Phase 2): We used `temperature=0.6`, so slight variation

### Q: How big are activation files?

**A:**
```python
# One layer, one question
[1536] floats × 4 bytes/float = 6 KB

# One layer, 30 faithful + 20 unfaithful
[50, 1536] floats × 4 bytes/float = 300 KB

# All 4 layers
300 KB × 4 layers = 1.2 MB

# Actual file size is slightly more due to PyTorch overhead
# Real size: ~2-3 MB per file
```

---

**Need more help?** Check:
- `ACTIVATION_CACHING_EXPLAINED.md` - More detailed walkthrough
- `src/mechanistic/cache_activations.py` - Actual implementation
- `PHASE3_README.md` - Usage guide

**Document Version:** 1.0



