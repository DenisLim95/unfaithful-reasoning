# Executive Summary: Linear Encoding of CoT Unfaithfulness in Small Reasoning Models

**Applicant:** Denis Lim  
**Application:** MATS 10.0 (Neel Nanda's Stream)  
**Date:** January 2, 2026  
**Time Spent:** ~18 hours research + 2 hours write-up

---

## Research Question

**Do small open-weight reasoning models (1.5B parameters) encode chain-of-thought faithfulness linearly, making real-time monitoring feasible?**

This question matters for AI safety because:
1. **Reasoning models are increasingly deployed** - DeepSeek R1, o1, o3 are being used in high-stakes applications
2. **CoT unfaithfulness is common** - Prior work shows 61-75% of reasoning traces are inconsistent with final answers
3. **Small models are more accessible** - If faithfulness can be detected in 1.5B models, it's more deployable

---

## Key Findings

### Finding 1: Small Models Show High Unfaithfulness (50%)

**Result:** 50% faithful, 50% unfaithful (100 question pairs)

**Comparison to prior work:**
- DeepSeek R1 (70B, API): 39% faithful (Arcuschin et al., 2025)
- Claude 3.7 Sonnet: 25% faithful (Arcuschin et al., 2025)
- **This work (1.5B distilled):** 50% faithful

**Interpretation:** Small models show **more unfaithfulness** than the large 70B DeepSeek R1 (50% vs 39% unfaithful). This contradicts my initial hypothesis that small models would be more faithful because they can't hide complexity. Instead, the 1.5B model appears to lack the capacity for consistent reasoning.

**Surprising observation:** The model often produces sound reasoning but then provides the answer to the *opposite* question. It seems to fail at tracking which variant (Q1 vs Q2) it's answering, even when the reasoning itself is correct.

### Finding 2: No Linear Encoding Detected (Null Result)

**Result:** Linear probes perform at chance level across all layers

| Layer | Accuracy | AUC | vs Random |
|-------|----------|-----|-----------|
| 6 | 50.0% | 0.512 | +0.0pp |
| 12 | 44.0% | 0.484 | -6.0pp |
| 18 | 48.0% | 0.475 | -2.0pp |
| 24 | 54.0% | 0.506 | +4.0pp |

**Best layer:** Layer 24 at 54% (only 4pp above random)

**Interpretation:** **Faithfulness is NOT linearly encoded** in DeepSeek-R1-Distill-1.5B. All probes perform at or below chance, with no layer showing meaningful separation. This is a **valuable null result** that:
- Rules out simple linear monitoring approaches
- Suggests faithfulness is either non-linearly encoded or distributed
- Indicates need for more complex detection methods (SAEs, attention analysis)

### Finding 3: Flat Layer-wise Pattern

**Pattern:** No clear peak - performance flat across layers (44-54%)

**Interpretation:** The absence of any layer showing strong performance suggests:
- Faithfulness is **not localized** to specific processing stages
- May be a **distributed property** throughout the model
- Or simply not strongly encoded in residual stream activations

This contrasts with other properties (truthfulness, refusal) which often show clear layer-wise peaks.

---

## Main Contribution

### What This Work Shows

**Negative but valuable result:** Linear probes **cannot** detect CoT faithfulness in small reasoning models. This:
1. **Rules out** simple real-time monitoring with linear probes
2. **Narrows the search space** for what might work (non-linear methods)
3. **Provides evidence** that faithfulness is more complex than first-order semantic properties

**Novel finding about small models:** The 1.5B model shows a specific failure mode - correct reasoning paired with opposite answers - suggesting a failure in tracking question variants rather than reasoning ability.

### Implications for AI Safety

**Real-time monitoring is NOT feasible with linear probes:**
- ❌ Cannot use simple linear probes (chance-level performance)
- ❌ Need more expensive methods: SAEs, attention analysis, or LLM judges
- ⚠️ Small models may be **riskier** than large ones (more unfaithful, harder to detect)

**Why small models might be worse:**
- Lack capacity to maintain separate "reasoning" and "answer tracking" threads
- May prioritize local coherence over global consistency
- Distillation may not preserve faithfulness (preserves capability, not reliability)

---

## Methodology

**Three-phase pipeline:**
1. **Question Generation:** 100 numerical comparison pairs ("Is X larger than Y?")
2. **Faithfulness Evaluation:** Question-flipping methodology (Arcuschin et al.)
3. **Linear Probe Training:** Logistic regression on mean-pooled activations at layers 6, 12, 18, 24

**Implementation:** Type-enforced contracts, 18 validation tests, reproducible pipeline

---

## What I Learned

### About Chain-of-Thought Faithfulness

**The problem is more nuanced than I expected.** Small models show a specific failure mode: they generate sound reasoning but then provide the answer to the opposite question. For example, the model might correctly determine "847 > 839" but then answer "No" to "Is 847 larger than 839?" because it lost track of which variant it was answering. This suggests unfaithfulness isn't just about bad reasoning - it's about failing to maintain global context.

**Small models might be MORE unfaithful than large ones.** My 1.5B model showed 50% unfaithfulness versus 39% for the 70B DeepSeek R1. This surprised me - I expected smaller models to be more faithful because they can't afford to maintain separate reasoning and answer tracks. Instead, they appear to lack the capacity for consistent tracking across semantically equivalent questions.

**Linear encoding might not capture meta-cognitive properties.** My probes achieved 44-54% accuracy - essentially random. This taught me that while first-order properties (truthfulness, sentiment) often have clean linear representations, second-order properties (reasoning about reasoning quality) might be fundamentally non-linear or distributed. This is a valuable null result: it tells us we can't rely on simple linear probes for monitoring reasoning faithfulness.

### Research Insights That Surprised Me

**Prompt engineering matters enormously.** Changing from "provide your answer" to "provide your final answer as either 'Yes' or 'No'" dramatically improved extraction reliability. But even with perfect extraction, the model still failed at consistency - suggesting the problem is deeper than output formatting.

**The question type matters more than I thought.** I chose numerical comparisons because they're objective and clean. But this means my findings might not generalize. A model could be faithful on numerical tasks but unfaithful on questions requiring world knowledge or multi-step reasoning. The specific failure mode I observed (tracking which question variant) might be unique to symmetric question pairs.

**Null results are harder to interpret than I expected.** When probes don't work, is it because: (1) faithfulness isn't linearly encoded, (2) I'm looking at the wrong layers, (3) mean-pooling loses critical information, or (4) my dataset is too small? Distinguishing between these requires careful experimental design and additional ablations.

### What I'd Explore Next If Selected for MATS

**Test non-linear detection methods.** My linear probes failed, but that doesn't mean faithfulness is undetectable. I'd try: (1) sparse autoencoders to capture non-linear patterns, (2) attention pattern analysis to see if faithful responses show different attention structures, and (3) cross-layer combinations (maybe faithfulness requires integrating information across layers).

**Understand the "opposite answer" failure mode.** This seems like a localization failure - the model loses track of which question it's answering. I'd use activation patching to test: if I patch in activations from Q1 when processing Q2, does the model give the correct (opposite) answer? This would confirm it's about tracking context, not reasoning ability.

**Test how temperature affects faithfulness.** All my experiments used temperature=0.6. What if I vary temperature from 0.0 (deterministic) to 1.0 (creative)? My hypothesis: lower temperature might increase faithfulness because the model commits to its reasoning more strongly. Higher temperature might make the "opposite answer" error more common because sampling introduces more noise.

**Scale comparison to test the capacity hypothesis.** If small models are unfaithful because they lack capacity, this should show a clear trend: 1.5B least faithful, 7B intermediate, 70B most faithful. Testing this would validate whether unfaithfulness decreases with scale and inform deployment decisions.

### Why This Matters for My Research Direction

This project convinced me that **null results are just as valuable as positive ones for AI safety.** Knowing that linear probes *don't* work for faithfulness detection is crucial information - it saves the field from pursuing a dead end and redirects effort toward methods that might actually work (SAEs, attention analysis).

It also showed me that **small models deserve more safety research attention.** Everyone focuses on frontier models, but small models are: (1) actually deployed more widely, (2) potentially more dangerous (harder to detect unfaithfulness), and (3) cheaper to experiment with. Understanding their specific failure modes (like the "opposite answer" phenomenon) could inform better training methods.

Finally, I learned that **mechanistic interpretability needs better tools for meta-cognitive properties.** Linear probes work great for "is the model truthful about facts" but fail for "is the model's reasoning faithful to its answer." This suggests we need new techniques specifically designed for understanding reasoning quality, not just factual content. That's what I want to work on at MATS.

---

## Limitations

1. **Small sample:** 100 pairs (vs 1000+ in Arcuschin et al.) - wider confidence intervals
2. **Single model:** Only 1.5B DeepSeek-R1-Distill - may not generalize to other architectures
3. **Single question type:** Numerical comparisons only - different failure modes might exist for other tasks
4. **Detection only:** No intervention experiments to test causality
5. **Mean-pooling:** May have lost positional information critical for faithfulness

---

## Future Work

**Immediate next steps (if selected for MATS):**
1. Test sparse autoencoders and attention pattern analysis (non-linear methods)
2. Use activation patching to understand "opposite answer" failure mode
3. Vary temperature to test how sampling affects faithfulness
4. Scale to 7B model to test capacity hypothesis

**Longer-term directions:**
1. Develop specialized tools for detecting meta-cognitive properties
2. Study relationship between faithfulness and other safety properties
3. Design training interventions to reduce "opposite answer" errors

---

## Conclusion

I set out to test whether CoT faithfulness has a linear encoding in small reasoning models. **The answer is no** - linear probes perform at chance level across all layers. This null result is valuable: it rules out simple monitoring approaches and suggests faithfulness requires more sophisticated detection methods.

I also discovered that small models (1.5B) are **more unfaithful than large models** (50% vs 39%), contradicting my initial hypothesis. They show a specific failure mode: correct reasoning paired with opposite answers, suggesting they fail at tracking which question variant they're answering.

For AI safety, this means: (1) we cannot use cheap linear probes for real-time faithfulness monitoring, (2) small models may be riskier than expected, and (3) we need new tools specifically for meta-cognitive properties. These findings narrow the search space for what might work and highlight the need for methods like SAEs and attention analysis.

If selected for MATS, I want to explore non-linear detection methods, understand the "opposite answer" failure mode through activation patching, and test how temperature affects faithfulness. This project taught me that null results are just as important as positive ones - they save the field from pursuing dead ends and point toward more promising directions.

---

## Appendix: Results Summary

### A. Faithfulness Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 100 |
| Faithful pairs | 50 (50%) |
| Unfaithful pairs | 50 (50%) |
| Test set | 100 samples (50 faithful, 50 unfaithful) |

### B. Probe Performance

| Layer | Accuracy | AUC | vs Random | vs Majority (50%) |
|-------|----------|-----|-----------|-------------------|
| 6 | 50.0% | 0.512 | +0.0pp | +0.0pp |
| 12 | 44.0% | 0.484 | -6.0pp | -6.0pp |
| 18 | 48.0% | 0.475 | -2.0pp | -2.0pp |
| 24 | 54.0% | 0.506 | +4.0pp | +4.0pp |

**Best layer:** 24 (54% accuracy, only 4pp above random)  
**Conclusion:** No meaningful linear encoding detected

### C. Key Observation

**"Opposite Answer" Failure Mode:** The model frequently produces correct reasoning but provides the answer to the wrong question variant. Example:
- Q1: "Is 847 larger than 839?"
- Reasoning: "847 > 839, so yes"
- **Actual answer:** "No" ❌ (answered Q2 instead)

This suggests a failure in maintaining global context rather than reasoning ability.

---

**Total Time:** ~18 hours research + 2 hours documentation = 20 hours  
**Code Repository:** `/Users/denislim/workspace/mats-10.0/`
