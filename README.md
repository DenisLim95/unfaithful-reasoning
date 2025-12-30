# CoT Unfaithfulness in Small Reasoning Models

Mechanistic analysis of chain-of-thought faithfulness in DeepSeek-R1-Distill models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. For GPU support (recommended)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Run Phase 1 (generate questions)
python src/data_generation/generate_questions.py

# 4. Run Phase 2 (faithfulness evaluation)
./run_phase2.sh
```

## Project Structure

```
├── requirements.txt              # All dependencies (all phases)
├── src/
│   ├── data_generation/         # Phase 1: Question generation
│   ├── inference/               # Phase 2: Model inference
│   ├── evaluation/              # Phase 2: Faithfulness scoring
│   └── mechanistic/             # Phase 3: Mechanistic analysis (TBD)
├── tests/                       # Validation scripts
├── data/                        # Generated data
└── results/                     # Analysis outputs
```

## Documentation

- **START_HERE.md** - Quick overview
- **PHASE2_QUICKSTART.md** - Phase 2 quick start guide
- **PHASE2_README.md** - Complete Phase 2 reference
- **technical_specification.md** - Full technical specification
- **phased_implementation_plan.md** - Implementation roadmap

## Dependencies

All dependencies are in `requirements.txt` (covers all phases).

**Key packages:**
- `torch==2.2.0` - Deep learning framework
- `transformers==4.39.0` - HuggingFace models
- `accelerate==0.27.0` - Model loading utilities
- `numpy>=1.26.0,<2.0.0` - Numerical operations (NumPy 1.x required)
- `pandas==2.2.0` - Data manipulation
- `transformer-lens==1.17.0` - Mechanistic interpretability (Phase 3)

## GPU Setup

For GPU support:
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU only (slower):
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

## Current Status

- ✅ **Phase 1:** Question generation complete
- ✅ **Phase 2:** Faithfulness evaluation implemented
- ⏳ **Phase 3:** Mechanistic analysis (TBD)
- ⏳ **Phase 4:** Report and analysis (TBD)

## Citation

Based on:
```
Arcuschin et al. (2025). "Reasoning Models Don't Always Say What They Think"
arXiv:2505.05410
```
