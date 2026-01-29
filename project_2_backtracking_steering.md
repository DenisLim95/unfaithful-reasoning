# Project 2: Backtracking Steering Vectors in Reasoning Models

## ⚠️ HIGH RISK WARNING

**This project has significant novelty concerns.** Two recent papers by Neel's own MATS scholars have extensively covered this territory:
1. Arcuschin & Venhoff (June 2025) - "Understanding Reasoning in Thinking Language Models via Steering Vectors"
2. Ward et al. (July 2025) - "Reasoning-Finetuning Repurposes Latent Representations in Base Models"

**Only pursue this project if you have a very specific novel angle not covered below.**

---

## Overview

**Core Question:** Can reasoning behaviors in thinking LLMs (like backtracking, uncertainty expression) be controlled via steering vectors, and what do these vectors represent?

**Why This Is Interesting:** Understanding how to control reasoning processes could enable safer AI systems—we could potentially detect or suppress problematic reasoning patterns.

**Alignment with Neel's Interests:**
- Neel co-authored the Ward et al. paper
- Listed "steering vectors for reasoning" in his research directions
- But: he's already seen extensive work here from recent scholars

---

## What's Already Been Done (EXTENSIVELY)

### Paper 1: Arcuschin & Venhoff (June 2025)
"Understanding Reasoning in Thinking Language Models via Steering Vectors"

**Models:** DeepSeek-R1-Distill models (multiple sizes)

**What they did:**
- Identified key reasoning behaviors: backtracking, uncertainty expression, example testing
- Extracted steering vectors for each behavior
- Showed these behaviors are mediated by linear directions in activation space
- Demonstrated bidirectional control (increase/decrease backtracking)
- Tested across 500 tasks in 10 categories

**Key findings:**
- Thinking models backtrack ~3x more than non-thinking models
- Steering vectors work consistently across model sizes
- Can modulate backtracking, uncertainty, and example generation

### Paper 2: Ward et al. (July 2025)
"Reasoning-Finetuning Repurposes Latent Representations in Base Models"

**Models:** DeepSeek-R1-Distill-Llama-8B and base Llama-3.1-8B

**What they did:**
- Found backtracking direction in the BASE model (Llama-3.1-8B)
- Showed this direction induces backtracking in the DISTILLED reasoning model
- Demonstrated >0.7 cosine similarity between base and reasoning model vectors
- Showed the effect isn't just token-level

**Key findings:**
- Reasoning capabilities are REPURPOSED from pre-existing representations
- The base model already has the "machinery" for backtracking
- RL fine-tuning activates/repurposes these latent directions

---

## What Might Still Be Novel (Narrow Gaps)

### Gap 1: Cross-Family Transfer
- Existing work: Llama-based distills
- Novel: Do Qwen-based distill steering vectors work on Llama-based distills (or vice versa)?
- Risk: Probably not that interesting if they do/don't transfer

### Gap 2: What Does the Vector Represent?
- Ward et al. noted the "uncertainty hypothesis" was inconclusive
- Novel: Deeper investigation into the semantic meaning of the backtracking direction
- Risk: This is hard to do rigorously in 20 hours

### Gap 3: Other Reasoning Behaviors
- Existing work: Backtracking, uncertainty, example testing
- Novel: Self-correction, hypothesis generation, verification steps
- Risk: Arcuschin already covered many behaviors

### Gap 4: Interaction Effects
- Novel: What happens when you steer multiple behaviors simultaneously?
- Risk: Complex to design, may not yield clear results

### Gap 5: Safety-Relevant Steering
- Novel: Can you use reasoning steering vectors to affect safety behaviors?
- E.g., does increasing backtracking make models more likely to catch their own mistakes on harmful queries?
- Risk: This pivots toward Project 3 territory

---

## If You Still Want to Pursue This

### The Only Viable Angle: Connect to Safety

Instead of pure backtracking analysis, study:
**"Does steering reasoning behaviors affect safety-relevant outputs?"**

Example research questions:
1. If you increase backtracking, does the model catch more of its own harmful outputs?
2. If you increase uncertainty expression, does the model refuse more appropriately?
3. Can reasoning steering vectors be used as a monitoring signal for problematic reasoning?

This connects the well-explored steering work to Neel's core focus on safety.

---

## 20-Hour Implementation Plan (If Pursuing)

### Hours 1-4: Replication Setup
- Load DeepSeek-R1-Distill-Qwen-7B
- Implement basic backtracking detection (look for "wait", "let me reconsider", "actually")
- Verify you can identify backtracking in model outputs

### Hours 5-8: Extract Steering Vectors
- Collect activations from backtracking vs non-backtracking responses
- Compute mean difference vectors (DiffMean approach)
- Test basic steering: does adding/subtracting the vector change backtracking rate?

### Hours 9-12: Novel Direction
Choose ONE:
- **Cross-family transfer:** Test Qwen-derived vectors on Llama distill
- **Safety connection:** Test if steering affects refusal/safety behaviors
- **Semantic probing:** What other behaviors correlate with backtracking direction?

### Hours 13-16: Quantitative Validation
- Measure effect sizes
- Compare to baselines (random vectors, other layer vectors)
- Statistical significance testing

### Hours 17-18: Sanity Checks
- Manual inspection of steered outputs
- Check for side effects (does steering break coherence?)

### Hours 19-20: Write-Up
- Frame findings relative to existing work
- Emphasize what's novel
- Be honest about limitations

---

## Models & Tools

### Recommended Models
| Model | Parameters | Notes |
|-------|------------|-------|
| DeepSeek-R1-Distill-Qwen-7B | 7B | Primary target |
| DeepSeek-R1-Distill-Llama-8B | 8B | For cross-family comparison |
| Llama-3.1-8B-Base | 8B | For base model comparison |

### Libraries
- **nnsight** - Recommended by Neel for larger models
- **TransformerLens** - If model is supported
- **baukit** - For activation patching

### Key Technique: DiffMean Steering
```python
# Pseudocode for steering vector extraction
backtracking_activations = collect_activations(backtracking_prompts)
non_backtracking_activations = collect_activations(non_backtracking_prompts)

steering_vector = mean(backtracking_activations) - mean(non_backtracking_activations)

# Apply steering
def steer(model, prompt, steering_vector, strength=1.0):
    # Add steering_vector * strength to residual stream at layer L
    pass
```

---

## Potential Pitfalls

### Pitfall 1: Results Match Existing Papers Exactly
**Likelihood: HIGH**

If your results just confirm Ward et al. or Arcuschin, you haven't added anything.

**Mitigation:** You MUST have a novel angle from the start. Don't just replicate.

### Pitfall 2: Steering Vectors Don't Work for You
Sometimes replication fails due to implementation details.

**Mitigation:** 
- Follow existing implementations closely
- Use same hyperparameters (layer selection, steering strength)
- If it fails, document why—that's also a finding

### Pitfall 3: Novel Angle Is Negative Result
E.g., "Cross-family transfer doesn't work"

**Mitigation:** Negative results are fine IF you:
- Had a clear hypothesis
- Tested it properly
- Explain why it matters that it failed

### Pitfall 4: Competition with Recent Scholars
Neel's recent MATS scholars literally just did this. They know more than you.

**Mitigation:** Don't try to out-do them. Find a perpendicular angle.

---

## What Success Looks Like

### Minimum Viable (Unlikely to Impress)
- Replication of steering vectors
- Same findings as existing papers
- Clean write-up

### Acceptable Project
- Novel angle that extends existing work
- Clear finding that wasn't in Ward/Arcuschin papers
- Honest positioning relative to prior work

### Strong Project (Hard to Achieve)
- Surprising finding about steering vectors
- Connection to safety that opens new research directions
- Something that makes Neel say "I didn't think of that"

---

## Honest Recommendation

**Don't do this project unless:**
1. You have a very specific novel angle not covered above
2. You're confident you can execute something beyond replication
3. You're willing to pivot if early results match existing work

**Better alternatives:**
- Project 3 (User Model Probing) has more open space
- Project 1 (CoT Unfaithfulness) has clearer novel angles

---

## Key Resources

### Papers to Read (You Must Understand These)
1. [Arcuschin & Venhoff - Understanding Reasoning via Steering Vectors](https://arxiv.org/abs/2506.18167)
2. [Ward et al. - Reasoning-Finetuning Repurposes Latent Representations](https://arxiv.org/abs/2507.12638)
3. [Rimsky et al. - Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) - General steering technique

### Existing Code
- Check if Ward et al. or Arcuschin released code (look on GitHub/papers with code)
- Steering vector implementations exist in many repos

---

## Novelty Risk Assessment

| Aspect | Risk Level | Notes |
|--------|------------|-------|
| Core methodology | HIGH | Already done extensively |
| Model choice | HIGH | Same models used in prior work |
| Findings overlap | VERY HIGH | Recent MATS scholars covered this |
| Competition | VERY HIGH | You're competing with Neel's own scholars |
| Novel angles | NARROW | Few unexplored gaps remain |

**Overall: HIGH RISK, UNCERTAIN REWARD**

This project is well-trodden ground. Only pursue if you have a genuinely novel angle that you're confident about. Otherwise, choose a different project.
