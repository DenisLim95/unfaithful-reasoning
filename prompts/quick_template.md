# Quick Task Prompt Template

```markdown
# Phase [X], Task [X.Y] - [Task Name]

## Context
**Phase:** [X] - [Phase Name]
**Task:** [Task ID and description]
**Files:** [file paths to create]
**Runs on:** [Local laptop / Remote GPU pod]

## Requirements
**Data Contract:** [Output schema/format]
**Must Pass:** `tests/validate_phase[X].py`
**Key Rules:** [2-3 critical constraints]

## Deliverables
1. Code for [files]
2. Run with: `[command]`
3. Verify with: `[validation command]`

Please implement following @phased_implementation_plan.md lines [X-Y].
```

**Note:** Code written locally, GPU tasks (Phase 2-3) run on remote pod. Use relative paths, no interactive prompts.

---

## Usage Examples

### Minimal (for simple tasks)
```
Phase 1, Task 1.2 - Generate Questions
Files: src/data_generation/generate_questions.py
Runs on: Local laptop (no GPU needed)
Output: 50 question pairs in data/raw/question_pairs.json
Must pass: tests/validate_questions.py
Implement from plan lines 149-235.
```

### Standard (most tasks)
```
Phase 2, Task 2.1 - Model Inference
Files: src/inference/batch_inference.py
Runs on: Remote GPU pod (will transfer via rsync)
Contract: 100 responses in JSONL format (plan lines 513-528)
Must pass: tests/validate_phase2.py
Key constraints: Use temp=0.6, extract <think> tags, handle 50 pairs × 2 variants, no interactive prompts
Implement from plan lines 624-768.
```

### Detailed (complex tasks)
```
Phase 3, Task 3.3 - Train Linear Probes
Files: src/mechanistic/train_probes.py
Contract: Probe results dict with accuracy, auc, direction per layer (plan lines 1326-1349)
Input: data/activations/layer_*_activations.pt (faithful/unfaithful tensors)
Output: results/probe_results/all_probe_results.pt + probe_performance.png
Must pass: tests/validate_phase3.py (accuracy > 0.5)
Key constraints:
- 80/20 train/test split, stratified
- BCEWithLogitsLoss, Adam lr=1e-3, 50 epochs
- Test layers [6, 12, 18, 24]
Implement from plan lines 1552-1673.
```

