# Quick Start: Layer Progression Visualizations

## 🎬 View the Animations

```bash
# Open the recommended combined visualization
open results/activation_visualizations/layer_progression_combined.gif

# Or view all 4 animations
open results/activation_visualizations/layer_progression_*.gif
```

## 📊 View the Static Plots

```bash
# Side-by-side comparison of all layers
open results/activation_visualizations/layer_comparison_grid.png

# Separation trend across layers
open results/activation_visualizations/separation_across_layers.png
```

## 🔄 Regenerate Visualizations

### Animations
```bash
# Default (all modes, 1 fps)
python animate_layer_progression.py --fps 1

# Faster animation
python animate_layer_progression.py --fps 4

# Only the combined view (recommended)
python animate_layer_progression.py --mode combined --fps 2

# Use test data instead
python animate_layer_progression.py --activations-dir data/test_activations
```

### Static Plots
```bash
# Generate grid and separation plot
python create_layer_comparison_grid.py

# Use test data
python create_layer_comparison_grid.py --activations-dir data/test_activations
```

## 🎯 Key Finding

**Separation increases dramatically from layer 18 → 24:**
- Layer 6: 0.287
- Layer 12: 0.377
- Layer 18: 0.414
- **Layer 24: 0.862** ⭐ (3x stronger than layer 6!)

This shows the model builds up faithfulness gradually, then makes a sudden "decision" in the final layers.

## 📖 Documentation

- **`README_ANIMATIONS.md`** - Detailed guide to each animation type
- **`ANIMATION_SUMMARY.md`** - Implementation overview
- **`VISUALIZATION_FINDINGS.md`** - Research insights and interpretation

## 🔬 Next Steps

1. **Watch the animations** - especially `layer_progression_combined.gif`
2. **Identify outliers** - which points behave unusually?
3. **Test on new data** - does the pattern hold?
4. **Focus on layer 24** - it has the strongest separation

## ❓ Quick FAQ

**Q: Why GIF instead of MP4?**
A: FFmpeg not installed. Install with `brew install ffmpeg` for better quality MP4s.

**Q: Can I change the speed?**
A: Yes! Use `--fps` flag. Higher = faster (try `--fps 4`).

**Q: Which visualization should I look at first?**
A: Start with `layer_progression_combined.gif` - it combines the most interpretable axis (probe direction) with variance-capturing PCA.

**Q: What do the colors mean?**
A: Blue = Faithful responses, Red = Unfaithful responses

**Q: What is "probe direction"?**
A: The direction in activation space that your linear probe learned to separate faithful from unfaithful. It's the "faithfulness axis."

## 🎨 Customization

```bash
# Different layers
python animate_layer_progression.py --layers 6 12 18 24

# Different activation directory
python animate_layer_progression.py --activations-dir data/test_activations

# Specific mode only
python animate_layer_progression.py --mode probe  # or pca, global_pca, combined, all
```

---

**TL;DR:** Run `open results/activation_visualizations/layer_progression_combined.gif` to see how faithful vs unfaithful activations evolve across layers!

