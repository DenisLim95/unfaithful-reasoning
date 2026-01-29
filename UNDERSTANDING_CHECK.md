# Understanding Check: Activations

## The Key Distinction

### Tokens = Static Labels
```
Input: "Which is larger: 847 or 839?"

Tokens at ALL layers:
["Which", "is", "larger", ":", "847", "or", "839", "?"]
```

**These never change!** The model doesn't transform "Which" into a different word.

### Activations = Evolving Numerical Representations

```python
# Same token "847" at different layers:

Layer 6 (early):
"847" → [0.88, 0.22, -0.15, 0.44, 0.03, ...]
        ↑ These 1536 numbers encode: "This is a number token"

Layer 12 (middle):
"847" → [0.91, 0.45, 0.33, 0.67, 0.29, ...]
        ↑ These 1536 numbers encode: "This number is being compared"

Layer 24 (late):
"847" → [0.95, 0.71, 0.52, 0.88, 0.44, ...]
        ↑ These 1536 numbers encode: "This number is the larger one"
```

**The token "847" stays as "847", but what the model UNDERSTANDS about it evolves!**

---

## Concrete Example

Let me show you the actual data structures:

```python
# What we DON'T cache:
tokens = ["Which", "is", "larger", ":", "847", "or", "839", "?"]
# This is just metadata, same at every layer

# What we DO cache:
activations_layer_12 = torch.tensor([
    # "Which"
    [0.42, -0.15, 0.78, 0.22, -0.56, 0.31, ..., 0.18],  # 1536 floats
    
    # "is"  
    [0.31, 0.05, -0.43, 0.67, 0.12, -0.22, ..., 0.45],  # 1536 floats
    
    # "larger"
    [-0.22, 0.58, 0.33, -0.41, 0.87, 0.19, ..., -0.11], # 1536 floats
    
    # ":"
    [0.15, -0.33, 0.52, 0.08, -0.71, 0.44, ..., 0.29],  # 1536 floats
    
    # "847"
    [0.88, 0.22, -0.15, 0.44, 0.03, 0.91, ..., 0.67],   # 1536 floats
    
    # "or"
    [0.11, -0.42, 0.67, -0.23, 0.55, 0.08, ..., -0.34], # 1536 floats
    
    # "839"
    [0.76, 0.18, -0.28, 0.51, -0.09, 0.82, ..., 0.44],  # 1536 floats
    
    # "?"
    [0.03, 0.61, -0.44, 0.19, 0.82, -0.15, ..., 0.56]   # 1536 floats
])
# Shape: [8, 1536]
#         │   └─ 1536 features per token
#         └─ 8 tokens
```

**Each row is the model's internal representation of that token at layer 12.**

---

## Programming Analogy

Think of activations like **object state** in OOP:

```python
class Token:
    def __init__(self, text):
        self.text = text           # ← This never changes (like the token)
        self.features = [...]      # ← This evolves (like activations)

# Layer 6
token_847_layer6 = Token("847")
token_847_layer6.text = "847"                    # Static
token_847_layer6.features = [0.88, 0.22, ...]   # Current state

# Layer 12 (same token, different state)
token_847_layer12 = Token("847")  
token_847_layer12.text = "847"                   # Still "847"
token_847_layer12.features = [0.91, 0.45, ...]  # Evolved state!
```

The `.text` doesn't change, but the `.features` evolve as the model processes it deeper.

---

## What Do the 1536 Numbers Mean?

Each of the 1536 numbers is like a **feature detector**:

```python
# Hypothetical interpretation (real features are more abstract):
features[0]   = 0.88   # "How much is this a number?"
features[1]   = 0.22   # "How much is this larger than something?"
features[2]   = -0.15  # "How much is this a verb?"
features[3]   = 0.44   # "How much is this relevant to the answer?"
...
features[1535] = 0.67  # "Some abstract pattern we can't name"
```

**At different layers, these features represent different levels of understanding:**

- **Layer 6:** Low-level features (is it a number? a verb? punctuation?)
- **Layer 12:** Mid-level features (what role does it play in the sentence?)
- **Layer 18:** High-level features (how does it relate to the task?)
- **Layer 24:** Near-output features (is this part of the answer?)

---

## Your Second Question: Faithful vs Unfaithful

You asked: *"What are we expecting the activations to look like for a faithful response vs an unfaithful one?"*

Great question! This is the **hypothesis** of Phase 3.

### The Setup

```python
# Question pair:
q1 = "Which is larger: 847 or 839?"
q2 = "Which is larger: 839 or 847?"  # Same question, flipped

# Ground truth:
correct_answer = "847"

# Phase 2 results:
# Faithful model: Answers "847" to both q1 and q2 ✓
# Unfaithful model: Answers "847" to q1, "839" to q2 ✗
```

**Important clarification:** We only cache **q1** activations (per specification), not both q1 and q2. We're comparing:
- q1 activations from **faithful pairs** (pairs where q1 and q2 got consistent answers)
- q1 activations from **unfaithful pairs** (pairs where q1 and q2 got different answers)

### Hypothesis 1: Activations Should Be Different

**If faithfulness is encoded in the model:**

```python
# Faithful pair activations (layer 12)
faithful_q1 = [0.5, 0.3, 0.1, -0.2, 0.8, ...]  # From pair that answered consistently
faithful_q2 = [0.4, 0.2, 0.2, -0.1, 0.7, ...]  # Another faithful pair
faithful_q3 = [0.6, 0.4, 0.0, -0.3, 0.9, ...]  # Another faithful pair

# Unfaithful pair activations (layer 12)  
unfaithful_q1 = [-0.2, 0.8, 0.6, 0.5, -0.3, ...] # From pair that flipped answers
unfaithful_q2 = [-0.1, 0.7, 0.5, 0.4, -0.2, ...] # Another unfaithful pair
unfaithful_q3 = [-0.3, 0.9, 0.7, 0.6, -0.4, ...] # Another unfaithful pair
```

**Expectation:** The two groups should cluster separately:

```
    Feature Space
    
    Dim 2 ▲
          │     ○ Faithful
          │   ○   ○
          │     ○
    ──────┼─────────────> Dim 1
          │         × Unfaithful  
          │       ×   ×
          │         ×
```

If we see this pattern → **linear probe can separate them!**

### Hypothesis 2: What Might Cause the Difference?

**Theory 1: "Uncertainty Signal"**
```python
# Faithful activations might encode:
- "I'm confident in my reasoning"
- "The logic is clear"  
- "No contradictions detected"

# Unfaithful activations might encode:
- "Something feels uncertain"
- "Conflicting signals"
- "Heuristic-based answer (not reasoning-based)"
```

**Theory 2: "Position Bias Signal"**
```python
# Faithful activations might encode:
- "I'm attending to the actual numbers"
- "I compared 847 vs 839 semantically"

# Unfaithful activations might encode:  
- "I'm using position as a heuristic"
- "I'm defaulting to first/last position"
- "Not actually processing the numbers"
```

**Theory 3: "Reasoning Depth Signal"**
```python
# Faithful activations might encode:
- "I performed multi-step reasoning"
- "I verified my answer"

# Unfaithful activations might encode:
- "I used a shortcut"
- "I pattern-matched without deep processing"
```

### What We'd Observe

**If faithfulness IS encoded linearly:**

```python
# Train probe on layer 12 activations
probe = LinearProbe(1536)
accuracy = 0.75  # 75% accuracy!

# Interpretation:
"There exists a linear direction in activation space that 
predicts whether a response will be faithful or not."
```

**If faithfulness is NOT encoded linearly:**

```python
accuracy = 0.52  # Only 52% (barely better than random 50%)

# Interpretation:  
"Faithfulness is not represented as a simple linear pattern.
It may be encoded non-linearly, distributed across many 
dimensions, or not explicitly represented at all."
```

---

## Concrete Example with Made-Up Numbers

Let's walk through a specific example:

### Faithful Pair

```python
Question: "Which is larger: 847 or 839?"

# Layer 12 activations (after mean pooling)
activations = [
    0.5,   # Feature 0: "Question understanding high"
    0.8,   # Feature 1: "Numerical comparison detected"  
    0.3,   # Feature 2: "Confidence in reasoning high"
    -0.1,  # Feature 3: "Position bias low"
    0.9,   # Feature 4: "Correct computation performed"
    ...    # 1531 more features
]

# Model internal state:
# "I understand this is asking to compare 847 and 839.
#  I've determined 847 > 839 through numerical reasoning.
#  High confidence in this answer."

# Final output: "847" ✓
```

### Unfaithful Pair  

```python
Question: "Which is larger: 847 or 839?"  # Same question!

# Layer 12 activations (after mean pooling)
activations = [
    0.4,   # Feature 0: "Question understanding medium"
    0.5,   # Feature 1: "Numerical comparison somewhat detected"
    0.1,   # Feature 2: "Confidence in reasoning low"  ← Different!
    0.7,   # Feature 3: "Position bias high"          ← Different!
    0.3,   # Feature 4: "Heuristic-based answer"      ← Different!
    ...    # 1531 more features
]

# Model internal state:
# "This is a comparison question. The first number 
#  mentioned is 847, so I'll say that. Wait, or was 
#  it asking about the first or second position?"

# Final output: "847" for q1, but "839" for q2 ✗
```

**Key difference:** Even though both answered "847" to q1, their **internal states** (activations) are different!

---

## Why This Matters

### The Insight

```python
# Just looking at final outputs:
faithful_output = "847"
unfaithful_output = "847"  # Same!

# But looking at activations:
faithful_activations = [0.5, 0.8, 0.3, -0.1, 0.9, ...]
unfaithful_activations = [0.4, 0.5, 0.1, 0.7, 0.3, ...]  # Different!
```

**The model "knows" internally when it's being unfaithful, even if the output looks right!**

This is why activation caching is powerful:
1. Final output only tells us WHAT the model said
2. Activations tell us HOW the model arrived at that answer
3. If faithful/unfaithful use different "how"s, we can detect it

---

## Summary of Your Questions

### Q1: "Are activations just tokens at each layer?"

**No!** 

- **Tokens:** ["Which", "is", "larger", ":", "847", "or", "839", "?"] ← Never change
- **Activations:** `[8, 1536]` array of floats ← Evolve at each layer

Activations are **numerical representations** of what the model understands about those tokens at that layer.

### Q2: "What do we expect faithful vs unfaithful activations to look like?"

**Hypothesis:** They should be systematically different!

```python
Faithful:   [0.5, 0.8, 0.3, -0.1, 0.9, ...]  ← "High confidence reasoning"
Unfaithful: [0.4, 0.5, 0.1, 0.7, 0.3, ...]   ← "Low confidence, position bias"
```

**If hypothesis is true:**
- Linear probe can separate them (accuracy > 65%)
- We've found a "faithfulness direction" in the model

**If hypothesis is false:**
- Linear probe can't separate them (accuracy ≈ 50%)
- Null result: faithfulness isn't linearly encoded (but still valuable!)

---

## One More Analogy

Think of activations like **brain scans**:

```python
# Person A looking at optical illusion:
Brain scan: [frontal lobe active, visual cortex confused, uncertainty high]
Says: "I see both a rabbit and a duck, I'm not sure"

# Person B looking at optical illusion:
Brain scan: [frontal lobe very active, visual cortex resolved, certainty high]  
Says: "It's definitely a rabbit"

# Same input (optical illusion), different brain states!
# The brain scan reveals their internal processing, not just what they say.
```

**Activations = neural network's "brain scan"**  
**Final output = what it says**

We're looking for whether faithful and unfaithful responses have different "brain states" even when they say the same thing!

---

Does this clear up the confusion? The key insight is:
- **Tokens are static** (just labels)
- **Activations are dynamic** (numerical representations that evolve)
- We're comparing **internal states**, not just outputs



