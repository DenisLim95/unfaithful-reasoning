"""
Technical Concepts Visualizer for SPAR Interview Prep

Generates educational diagrams explaining:
1. Activations - What they look like at each layer
2. Residual Stream - Information flow in transformers
3. Mean Pooling - Collapsing sequence dimension
4. Linear Probe - Classification from activations
5. Faithfulness Direction - Separation in activation space
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
import os

# Create output directory
OUTPUT_DIR = "results/technical_concept_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def visualize_activations():
    """
    Visualize what activations look like - the intermediate values at each layer.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("What Are Activations?", fontsize=16, fontweight='bold')
    
    # Panel 1: Input to activation concept
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("1. Input Text → Tokens → Numbers", fontsize=12)
    
    # Input text
    ax1.text(5, 9, '"Is 469 larger than 800?"', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue'))
    
    # Arrow
    ax1.annotate('', xy=(5, 7.5), xytext=(5, 8.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.text(5.5, 7.9, 'tokenize', fontsize=9, color='gray')
    
    # Tokens
    tokens = ['Is', '469', 'larger', 'than', '800', '?']
    for i, token in enumerate(tokens):
        x = 1.5 + i * 1.3
        ax1.add_patch(FancyBboxPatch((x-0.5, 6.3), 1, 0.8, boxstyle="round,pad=0.05",
                                      facecolor='lightyellow', edgecolor='orange'))
        ax1.text(x, 6.7, token, ha='center', va='center', fontsize=9)
    
    # Arrow
    ax1.annotate('', xy=(5, 5.5), xytext=(5, 6.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.text(5.5, 5.8, 'embed', fontsize=9, color='gray')
    
    # Activation matrix
    ax1.text(5, 4.8, 'Activations (numbers!)', ha='center', fontsize=10, fontweight='bold')
    
    # Small matrix visualization
    np.random.seed(42)
    data = np.random.randn(6, 8) * 0.5
    for i in range(6):
        for j in range(8):
            color = plt.cm.RdBu(0.5 + data[i, j] * 0.3)
            ax1.add_patch(Rectangle((1.5 + j*0.8, 1.5 + (5-i)*0.5), 0.75, 0.45,
                                     facecolor=color, edgecolor='gray', linewidth=0.5))
    
    ax1.text(0.8, 3.5, '6\ntokens', ha='center', va='center', fontsize=8)
    ax1.text(5, 1.0, '← 1536 features per token →', ha='center', fontsize=9)
    
    # Panel 2: Layer progression
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("2. Activations Change at Each Layer", fontsize=12)
    
    layers = ['Layer 0\n(raw)', 'Layer 6\n(syntax)', 'Layer 12\n(meaning)', 
              'Layer 18\n(reasoning)', 'Layer 24\n(decision)']
    colors = ['#ffcccc', '#ffddaa', '#ffffaa', '#aaffaa', '#aaddff']
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        y = 8 - i * 1.6
        ax2.add_patch(FancyBboxPatch((1, y-0.5), 8, 1.2, boxstyle="round,pad=0.1",
                                      facecolor=color, edgecolor='gray'))
        ax2.text(5, y, layer, ha='center', va='center', fontsize=10)
        
        # Small activation preview
        np.random.seed(42 + i * 10)
        for j in range(6):
            val = np.random.randn() * 0.3
            c = plt.cm.RdBu(0.5 + val)
            ax2.add_patch(Rectangle((7.5 + j*0.2, y-0.2), 0.18, 0.4, facecolor=c, edgecolor='none'))
        
        if i < 4:
            ax2.annotate('', xy=(5, y-0.7), xytext=(5, y-0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax2.text(5, 0.3, 'Each layer transforms the activations\n(refining the representation)', 
             ha='center', fontsize=9, style='italic')
    
    # Panel 3: What the numbers mean
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title("3. Each Number = A 'Feature'", fontsize=12)
    
    ax3.text(5, 9, 'activation[i] = how much feature i is present', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lavender'))
    
    features = [
        ('feature[0] = 0.8', '"comparison" concept', '#90EE90'),
        ('feature[1] = -0.2', '"negation" concept', '#FFB6C1'),
        ('feature[2] = 0.5', '"number" concept', '#90EE90'),
        ('feature[127] = 0.9', '"larger than" relation', '#90EE90'),
        ('...', '...', 'white'),
        ('feature[1535] = 0.1', 'some abstract feature', '#FFFACD'),
    ]
    
    for i, (feat, meaning, color) in enumerate(features):
        y = 7.5 - i * 1.1
        ax3.add_patch(FancyBboxPatch((0.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor='gray'))
        ax3.text(2.5, y, feat, ha='center', va='center', fontsize=9, family='monospace')
        ax3.text(7, y, meaning, ha='center', va='center', fontsize=9, style='italic')
    
    ax3.text(5, 0.5, 'We don\'t know exactly what each feature means,\nbut together they encode the model\'s "understanding"',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_what_are_activations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/01_what_are_activations.png")


def visualize_residual_stream():
    """
    Visualize the residual stream - the main highway of information flow.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("The Residual Stream: Information Highway in Transformers", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Main highway (residual stream)
    highway_y = 5
    ax.add_patch(FancyBboxPatch((0.5, highway_y-0.4), 13, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#E6F3FF', edgecolor='#0066CC', linewidth=3))
    ax.text(7, highway_y, 'RESIDUAL STREAM (main information flow)', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0066CC')
    
    # Layers as side branches
    layer_positions = [2, 5, 8, 11]
    layer_names = ['Layer 6', 'Layer 12', 'Layer 18', 'Layer 24']
    
    for i, (x, name) in enumerate(zip(layer_positions, layer_names)):
        # Attention branch (going up)
        ax.annotate('', xy=(x, highway_y + 0.5), xytext=(x, highway_y + 2),
                   arrowprops=dict(arrowstyle='<->', color='#FF6B6B', lw=2))
        ax.add_patch(FancyBboxPatch((x-0.8, highway_y + 2), 1.6, 0.8, boxstyle="round,pad=0.05",
                                     facecolor='#FFCCCC', edgecolor='#FF6B6B'))
        ax.text(x, highway_y + 2.4, 'Attention', ha='center', va='center', fontsize=9)
        
        # FFN branch (going down)
        ax.annotate('', xy=(x, highway_y - 0.5), xytext=(x, highway_y - 2),
                   arrowprops=dict(arrowstyle='<->', color='#4ECDC4', lw=2))
        ax.add_patch(FancyBboxPatch((x-0.8, highway_y - 2.8), 1.6, 0.8, boxstyle="round,pad=0.05",
                                     facecolor='#CCFFEE', edgecolor='#4ECDC4'))
        ax.text(x, highway_y - 2.4, 'FFN', ha='center', va='center', fontsize=9)
        
        # Layer label
        ax.text(x, highway_y + 3.5, name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plus signs showing residual addition
        ax.text(x + 0.9, highway_y + 1.2, '+', fontsize=14, fontweight='bold', color='green')
        ax.text(x + 0.9, highway_y - 1.2, '+', fontsize=14, fontweight='bold', color='green')
    
    # Input and output
    ax.add_patch(FancyBboxPatch((0, highway_y-0.3), 0.4, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='lightgreen', edgecolor='green'))
    ax.text(-0.3, highway_y, 'IN', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((13.6, highway_y-0.3), 0.4, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='lightcoral', edgecolor='red'))
    ax.text(14.1, highway_y, 'OUT', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows showing flow direction
    for x in [3.5, 6.5, 9.5]:
        ax.annotate('', xy=(x+0.5, highway_y), xytext=(x, highway_y),
                   arrowprops=dict(arrowstyle='->', color='#0066CC', lw=2))
    
    # Explanation text boxes
    ax.text(7, 8.5, 'Key Insight: Information ACCUMULATES', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    ax.text(7, 7.5, 'residual = residual + attention_output + ffn_output', 
            ha='center', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Why we cache residual stream
    ax.text(7, 1.2, 'Why cache the residual stream?', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 0.6, '• Contains ALL accumulated information\n• Most comprehensive view of model state\n• Standard in interpretability research',
            ha='center', fontsize=10, va='top')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_residual_stream.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/02_residual_stream.png")


def visualize_mean_pooling():
    """
    Visualize mean pooling - collapsing the sequence dimension.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Mean Pooling: From Variable-Length to Fixed-Size", fontsize=16, fontweight='bold')
    
    # Panel 1: The problem
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("The Problem", fontsize=12, color='red')
    
    ax1.text(5, 9, 'Different questions = different lengths!', ha='center', fontsize=11)
    
    # Question 1 (short)
    ax1.text(1, 7.5, 'Q1:', fontsize=10, fontweight='bold')
    ax1.text(2, 7.5, '"Is 469 > 800?"', fontsize=9)
    ax1.text(7, 7.5, '→ 6 tokens', fontsize=9, color='blue')
    
    # Draw 6-token matrix
    for i in range(6):
        for j in range(4):
            ax1.add_patch(Rectangle((2 + i*0.5, 6.2 + j*0.25), 0.45, 0.2,
                                     facecolor=plt.cm.Blues(0.3 + np.random.rand()*0.4),
                                     edgecolor='gray', linewidth=0.5))
    ax1.text(1, 6.7, 'Shape:\n[6, 1536]', fontsize=8, ha='center')
    
    # Question 2 (long)
    ax1.text(1, 4.5, 'Q2:', fontsize=10, fontweight='bold')
    ax1.text(2, 4.5, '"Compare 469 and 800, which is bigger?"', fontsize=9)
    ax1.text(8.5, 4.5, '→ 10 tokens', fontsize=9, color='blue')
    
    # Draw 10-token matrix
    for i in range(10):
        for j in range(4):
            ax1.add_patch(Rectangle((1 + i*0.5, 3.2 + j*0.25), 0.45, 0.2,
                                     facecolor=plt.cm.Oranges(0.3 + np.random.rand()*0.4),
                                     edgecolor='gray', linewidth=0.5))
    ax1.text(0.2, 3.7, 'Shape:\n[10, 1536]', fontsize=8, ha='center')
    
    ax1.text(5, 1.5, '❌ Can\'t stack different shapes!', ha='center', fontsize=11, 
             color='red', fontweight='bold')
    ax1.text(5, 0.7, 'Linear probe needs fixed input size', ha='center', fontsize=10)
    
    # Panel 2: The solution
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("The Solution: Mean Pooling", fontsize=12, color='green')
    
    ax2.text(5, 9, 'Average across all tokens!', ha='center', fontsize=11, fontweight='bold')
    
    # Before: matrix
    np.random.seed(123)
    ax2.text(2.5, 7.8, 'Before: [6, 1536]', ha='center', fontsize=10)
    data = np.random.randn(6, 8) * 0.3
    for i in range(6):
        for j in range(8):
            color = plt.cm.RdBu(0.5 + data[i, j])
            ax2.add_patch(Rectangle((0.5 + j*0.5, 5.5 + (5-i)*0.35), 0.45, 0.3,
                                     facecolor=color, edgecolor='gray', linewidth=0.5))
    
    # Arrow with "mean"
    ax2.annotate('', xy=(5, 4.8), xytext=(5, 5.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.text(5.8, 5.1, 'mean(dim=0)', fontsize=10, color='green', fontweight='bold')
    
    # After: vector
    ax2.text(2.5, 4.3, 'After: [1536]', ha='center', fontsize=10)
    means = data.mean(axis=0)
    for j in range(8):
        color = plt.cm.RdBu(0.5 + means[j])
        ax2.add_patch(Rectangle((0.5 + j*0.5, 3.5), 0.45, 0.5,
                                 facecolor=color, edgecolor='green', linewidth=2))
    
    # Formula
    ax2.text(5, 2.5, 'Each feature = average across all tokens', ha='center', fontsize=10)
    ax2.text(5, 1.8, 'pooled[j] = (token1[j] + token2[j] + ... + tokenN[j]) / N', 
             ha='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax2.text(5, 0.7, '✓ Now all questions have shape [1536]!', ha='center', fontsize=11,
             color='green', fontweight='bold')
    
    # Panel 3: Visual analogy
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title("Analogy: Team Average Score", fontsize=12)
    
    ax3.text(5, 9, 'Like calculating a team\'s average!', ha='center', fontsize=11)
    
    # Game scores
    games = [('Game 1', 85, '#FFB6C1'), ('Game 2', 92, '#98FB98'), 
             ('Game 3', 78, '#FFB6C1'), ('Game 4', 90, '#98FB98')]
    
    for i, (game, score, color) in enumerate(games):
        y = 7.5 - i * 1.2
        ax3.add_patch(FancyBboxPatch((1, y-0.4), 3, 0.8, boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor='gray'))
        ax3.text(2.5, y, f'{game}: {score}', ha='center', va='center', fontsize=10)
    
    # Arrow
    ax3.annotate('', xy=(5.5, 5), xytext=(5.5, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.text(6.2, 4, 'average', fontsize=10, color='blue')
    
    # Result
    ax3.add_patch(FancyBboxPatch((4, 1.8), 4, 1, boxstyle="round,pad=0.1",
                                  facecolor='#87CEEB', edgecolor='blue', linewidth=2))
    ax3.text(6, 2.3, 'Team Average: 86.25', ha='center', va='center', 
             fontsize=11, fontweight='bold')
    
    ax3.text(5, 0.7, 'One number summarizes the whole team\n= One vector summarizes all tokens',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_mean_pooling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/03_mean_pooling.png")


def visualize_linear_probe():
    """
    Visualize how a linear probe works and what the faithfulness direction means.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Linear Probe: Finding the 'Faithfulness Direction'", fontsize=16, fontweight='bold')
    
    # Panel 1: What is a linear probe
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("What is a Linear Probe?", fontsize=12)
    
    # Input activations
    ax1.text(2, 9, 'Input: Activations [1536]', ha='center', fontsize=10, fontweight='bold')
    
    np.random.seed(42)
    for i in range(10):
        val = np.random.randn() * 0.3
        color = plt.cm.RdBu(0.5 + val)
        ax1.add_patch(Rectangle((0.5 + i*0.35, 7.8), 0.3, 0.8,
                                 facecolor=color, edgecolor='gray', linewidth=0.5))
    ax1.text(4.5, 8.2, '...', fontsize=14)
    
    # Arrow to linear layer
    ax1.annotate('', xy=(2, 7), xytext=(2, 7.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Linear layer box
    ax1.add_patch(FancyBboxPatch((0.5, 5.5), 8, 1.3, boxstyle="round,pad=0.1",
                                  facecolor='#E8E8E8', edgecolor='black', linewidth=2))
    ax1.text(4.5, 6.5, 'Linear Layer', ha='center', fontsize=11, fontweight='bold')
    ax1.text(4.5, 5.9, 'output = weights · input + bias', ha='center', fontsize=9, family='monospace')
    
    # Weight vector
    ax1.text(7, 7.2, 'Weights:', fontsize=9)
    for i in range(6):
        ax1.add_patch(Rectangle((7 + i*0.25, 6.8), 0.2, 0.3,
                                 facecolor=plt.cm.Greens(0.4 + i*0.1), edgecolor='gray'))
    ax1.text(8.5, 6.5, '← This IS the\n"direction"!', fontsize=8, color='green')
    
    # Arrow to output
    ax1.annotate('', xy=(4.5, 4.8), xytext=(4.5, 5.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Output
    ax1.add_patch(FancyBboxPatch((3, 3.8), 3, 0.8, boxstyle="round,pad=0.1",
                                  facecolor='#98FB98', edgecolor='green', linewidth=2))
    ax1.text(4.5, 4.2, 'Output: 0.73', ha='center', fontsize=11, fontweight='bold')
    
    # Sigmoid
    ax1.annotate('', xy=(4.5, 2.8), xytext=(4.5, 3.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.text(5.2, 3.2, 'sigmoid', fontsize=9)
    
    # Probability
    ax1.add_patch(FancyBboxPatch((2, 1.8), 5, 0.8, boxstyle="round,pad=0.1",
                                  facecolor='#87CEEB', edgecolor='blue', linewidth=2))
    ax1.text(4.5, 2.2, 'P(faithful) = 0.67', ha='center', fontsize=11, fontweight='bold')
    
    ax1.text(4.5, 0.8, '> 0.5 → Predict "Faithful"', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Panel 2: The faithfulness direction
    ax2 = axes[1]
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title("The 'Faithfulness Direction' in 2D", fontsize=12)
    ax2.set_xlabel("Feature 1 (simplified)", fontsize=10)
    ax2.set_ylabel("Feature 2 (simplified)", fontsize=10)
    ax2.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Generate fake data
    np.random.seed(42)
    faithful_x = np.random.randn(15) * 0.5 + 1.2
    faithful_y = np.random.randn(15) * 0.5 + 0.8
    unfaithful_x = np.random.randn(12) * 0.5 - 0.8
    unfaithful_y = np.random.randn(12) * 0.5 - 0.5
    
    # Plot points
    ax2.scatter(faithful_x, faithful_y, c='blue', s=80, alpha=0.7, label='Faithful', edgecolors='darkblue')
    ax2.scatter(unfaithful_x, unfaithful_y, c='red', s=80, alpha=0.7, label='Unfaithful', edgecolors='darkred')
    
    # Draw the direction vector
    ax2.annotate('', xy=(1.8, 1.2), xytext=(-1.2, -0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.text(0.8, -0.3, 'Faithfulness\nDirection', fontsize=10, color='green', fontweight='bold',
             ha='center', rotation=30)
    
    # Draw decision boundary (perpendicular to direction)
    ax2.plot([-2, 2], [1.5, -1.5], 'k--', linewidth=2, alpha=0.5, label='Decision Boundary')
    
    # Annotations
    ax2.annotate('Faithful\nregion', xy=(1.5, 1.8), fontsize=10, ha='center', color='blue')
    ax2.annotate('Unfaithful\nregion', xy=(-1.5, -1.8), fontsize=10, ha='center', color='red')
    
    ax2.legend(loc='upper left', fontsize=9)
    
    # Add explanation box
    ax2.text(0, -2.7, 'The probe learns a direction that best separates the classes.\n'
                      'Projecting activations onto this direction gives a "faithfulness score".',
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_linear_probe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/04_linear_probe.png")


def visualize_faithfulness_concept():
    """
    Visualize faithful vs unfaithful chain-of-thought.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Faithful vs Unfaithful Chain-of-Thought", fontsize=16, fontweight='bold')
    
    # Panel 1: Faithful
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("✓ FAITHFUL Response", fontsize=14, color='green')
    
    # Question
    ax1.add_patch(FancyBboxPatch((0.5, 8.5), 9, 1, boxstyle="round,pad=0.1",
                                  facecolor='#E6F3FF', edgecolor='blue'))
    ax1.text(5, 9, 'Q: "Is 469 larger than 800?"', ha='center', va='center', fontsize=11)
    
    # Reasoning (genuine)
    ax1.add_patch(FancyBboxPatch((0.5, 4.5), 9, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#E8FFE8', edgecolor='green'))
    ax1.text(5, 7.5, '<think>', ha='center', fontsize=10, family='monospace', color='gray')
    ax1.text(5, 6.8, 'Let me compare these numbers...', ha='center', fontsize=10)
    ax1.text(5, 6.2, '469 has 3 digits, 800 has 3 digits', ha='center', fontsize=10)
    ax1.text(5, 5.6, '469 < 800 (comparing hundreds place)', ha='center', fontsize=10, fontweight='bold')
    ax1.text(5, 5.0, '</think>', ha='center', fontsize=10, family='monospace', color='gray')
    
    # Answer
    ax1.add_patch(FancyBboxPatch((3, 3), 4, 1, boxstyle="round,pad=0.1",
                                  facecolor='#98FB98', edgecolor='green', linewidth=2))
    ax1.text(5, 3.5, 'Answer: No', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Explanation
    ax1.text(5, 1.8, 'The reasoning ACTUALLY LED to the answer', ha='center', fontsize=11, 
             color='green', fontweight='bold')
    ax1.text(5, 1.0, 'The model genuinely compared the numbers\nand concluded 469 < 800', 
             ha='center', fontsize=10)
    
    # Arrow showing reasoning → answer
    ax1.annotate('', xy=(5, 3.2), xytext=(5, 4.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(5.5, 3.8, 'leads to', fontsize=9, color='green')
    
    # Panel 2: Unfaithful
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("✗ UNFAITHFUL Response", fontsize=14, color='red')
    
    # Question
    ax2.add_patch(FancyBboxPatch((0.5, 8.5), 9, 1, boxstyle="round,pad=0.1",
                                  facecolor='#E6F3FF', edgecolor='blue'))
    ax2.text(5, 9, 'Q: "Is 469 larger than 800?"', ha='center', va='center', fontsize=11)
    
    # Hidden decision (happens first!)
    ax2.add_patch(FancyBboxPatch((6.5, 7.2), 3, 1, boxstyle="round,pad=0.1",
                                  facecolor='#FFE0E0', edgecolor='red', linestyle='--'))
    ax2.text(8, 7.7, '(Decides "Yes"\nfirst!)', ha='center', fontsize=9, color='red', style='italic')
    
    # Reasoning (post-hoc rationalization)
    ax2.add_patch(FancyBboxPatch((0.5, 4.5), 9, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#FFE8E8', edgecolor='red'))
    ax2.text(5, 6.5, '<think>', ha='center', fontsize=10, family='monospace', color='gray')
    ax2.text(5, 5.9, 'Looking at these numbers...', ha='center', fontsize=10)
    ax2.text(5, 5.3, '469 appears to be the larger value...', ha='center', fontsize=10, 
             color='red', style='italic')
    ax2.text(5, 4.7, '</think>', ha='center', fontsize=10, family='monospace', color='gray')
    
    # Answer
    ax2.add_patch(FancyBboxPatch((3, 3), 4, 1, boxstyle="round,pad=0.1",
                                  facecolor='#FFB6C1', edgecolor='red', linewidth=2))
    ax2.text(5, 3.5, 'Answer: Yes ✗', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Explanation
    ax2.text(5, 1.8, 'The reasoning is POST-HOC RATIONALIZATION', ha='center', fontsize=11,
             color='red', fontweight='bold')
    ax2.text(5, 1.0, 'The model decided the answer FIRST,\nthen made up plausible-sounding reasoning',
             ha='center', fontsize=10)
    
    # Arrow showing answer → reasoning (backwards!)
    ax2.annotate('', xy=(6.5, 5.5), xytext=(7.5, 7.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    ax2.text(7.5, 6.3, 'justifies\n(backwards!)', fontsize=9, color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_faithful_vs_unfaithful.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/05_faithful_vs_unfaithful.png")


def visualize_layer_progression():
    """
    Visualize how separation between faithful/unfaithful emerges across layers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("How Faithful vs Unfaithful Separate Across Layers", fontsize=16, fontweight='bold')
    
    layers = [6, 12, 18, 24]
    # Simulate increasing separation
    separations = [0.3, 0.8, 0.9, 0.7]
    
    np.random.seed(42)
    
    for idx, (ax, layer, sep) in enumerate(zip(axes.flat, layers, separations)):
        # Generate data with increasing separation
        faithful_x = np.random.randn(15) * 0.6 + sep
        faithful_y = np.random.randn(15) * 0.6 + sep * 0.5
        unfaithful_x = np.random.randn(12) * 0.6 - sep
        unfaithful_y = np.random.randn(12) * 0.6 - sep * 0.5
        
        ax.scatter(faithful_x, faithful_y, c='blue', s=60, alpha=0.7, 
                   label='Faithful', edgecolors='darkblue')
        ax.scatter(unfaithful_x, unfaithful_y, c='red', s=60, alpha=0.7,
                   label='Unfaithful', edgecolors='darkred')
        
        # Mark centroids
        ax.scatter([np.mean(faithful_x)], [np.mean(faithful_y)], c='blue', s=200,
                   marker='*', edgecolors='black', linewidths=2, zorder=5)
        ax.scatter([np.mean(unfaithful_x)], [np.mean(unfaithful_y)], c='red', s=200,
                   marker='*', edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add separation metric
        distance = np.sqrt((np.mean(faithful_x) - np.mean(unfaithful_x))**2 + 
                          (np.mean(faithful_y) - np.mean(unfaithful_y))**2)
        ax.text(0.05, 0.95, f'Separation: {distance:.2f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9)
    
    # Add interpretation
    plt.figtext(0.5, 0.02, 
                'Key Insight: Middle layers (12, 18) show the best separation → '
                'this is where "faithfulness" is computed!\n'
                '★ = centroid (average position) of each class',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/06_layer_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/06_layer_progression.png")


def visualize_full_pipeline():
    """
    Visualize the complete pipeline from question to probe prediction.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Complete Pipeline: From Question to Faithfulness Prediction", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Step 1: Input question
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E6F3FF', edgecolor='blue', linewidth=2))
    ax.text(2, 8.7, '1. Question', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 8.1, '"Is 469 > 800?"', ha='center', fontsize=9)
    
    ax.annotate('', xy=(3.7, 8.25), xytext=(3.5, 8.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Step 2: Model generates response
    ax.add_patch(FancyBboxPatch((4, 7), 3.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFE8CC', edgecolor='orange', linewidth=2))
    ax.text(5.75, 9.2, '2. Model Response', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.75, 8.5, '<think>reasoning</think>', ha='center', fontsize=8, family='monospace')
    ax.text(5.75, 8.0, 'Answer: No', ha='center', fontsize=9)
    ax.text(5.75, 7.4, '+ Activations cached!', ha='center', fontsize=8, color='green')
    
    ax.annotate('', xy=(7.7, 8.25), xytext=(7.5, 8.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Step 3: Extract activations
    ax.add_patch(FancyBboxPatch((8, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E8FFE8', edgecolor='green', linewidth=2))
    ax.text(9.5, 8.7, '3. Get Activations', ha='center', fontsize=10, fontweight='bold')
    ax.text(9.5, 8.1, 'Layer 12: [1, seq, 1536]', ha='center', fontsize=8, family='monospace')
    
    ax.annotate('', xy=(11.2, 8.25), xytext=(11, 8.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Step 4: Mean pool
    ax.add_patch(FancyBboxPatch((11.5, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFFFCC', edgecolor='#CCCC00', linewidth=2))
    ax.text(12.75, 8.7, '4. Mean Pool', ha='center', fontsize=10, fontweight='bold')
    ax.text(12.75, 8.1, '[1536] vector', ha='center', fontsize=8, family='monospace')
    
    # Arrow down
    ax.annotate('', xy=(12.75, 7.3), xytext=(12.75, 7.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Step 5: Linear probe
    ax.add_patch(FancyBboxPatch((11, 5.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E8E8FF', edgecolor='purple', linewidth=2))
    ax.text(12.75, 6.7, '5. Linear Probe', ha='center', fontsize=10, fontweight='bold')
    ax.text(12.75, 6.1, 'w · x + b = logit', ha='center', fontsize=8, family='monospace')
    
    ax.annotate('', xy=(12.75, 5.3), xytext=(12.75, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Step 6: Prediction
    ax.add_patch(FancyBboxPatch((11.5, 4), 2.5, 1, boxstyle="round,pad=0.1",
                                 facecolor='#98FB98', edgecolor='green', linewidth=3))
    ax.text(12.75, 4.5, 'Faithful: 73%', ha='center', fontsize=11, fontweight='bold')
    
    # Side panel: What the probe learned
    ax.add_patch(FancyBboxPatch((0.5, 1), 6, 4, boxstyle="round,pad=0.1",
                                 facecolor='#F5F5F5', edgecolor='gray', linewidth=1))
    ax.text(3.5, 4.7, 'What the Probe Learned', ha='center', fontsize=11, fontweight='bold')
    
    # Training data
    ax.text(1, 4.0, 'Training Data:', fontsize=9, fontweight='bold')
    ax.text(1, 3.5, '• 30 faithful responses', fontsize=9)
    ax.text(1, 3.1, '• 14 unfaithful responses', fontsize=9)
    ax.text(1, 2.5, 'Learned:', fontsize=9, fontweight='bold')
    ax.text(1, 2.0, '• Weight vector (1536 dims)', fontsize=9)
    ax.text(1, 1.6, '• = "faithfulness direction"', fontsize=9, color='green')
    
    # Side panel: Why this works
    ax.add_patch(FancyBboxPatch((7, 1), 7, 4, boxstyle="round,pad=0.1",
                                 facecolor='#FFF5F5', edgecolor='gray', linewidth=1))
    ax.text(10.5, 4.7, 'Why This Works (If It Does)', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(7.5, 4.0, 'If the probe succeeds:', fontsize=9, fontweight='bold')
    ax.text(7.5, 3.5, '→ Faithfulness is linearly represented', fontsize=9)
    ax.text(7.5, 3.1, '→ There\'s a direction in activation space', fontsize=9)
    ax.text(7.5, 2.7, '   that correlates with faithfulness', fontsize=9)
    
    ax.text(7.5, 2.0, 'If the probe fails (accuracy ≈ 50%):', fontsize=9, fontweight='bold')
    ax.text(7.5, 1.5, '→ Either faithfulness isn\'t linearly encoded', fontsize=9)
    ax.text(7.5, 1.1, '→ Or we need more data/different approach', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_full_pipeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/07_full_pipeline.png")


def create_index_html():
    """Create an index HTML file for easy viewing."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Technical Concepts Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        img { max-width: 100%; border: 1px solid #ddd; margin: 20px 0; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
        .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Technical Concepts Visualizations</h1>
    <p>Visual explanations of key concepts from the CoT Unfaithfulness Detection project.</p>
    
    <h2>1. What Are Activations?</h2>
    <div class="description">
        Activations are the intermediate numerical values computed at each layer of the neural network.
        They represent the model's "understanding" at each processing stage.
    </div>
    <img src="01_what_are_activations.png" alt="What Are Activations">
    
    <h2>2. The Residual Stream</h2>
    <div class="description">
        The residual stream is the main "highway" of information flow in a transformer.
        Each layer adds to (rather than replaces) the information flowing through.
    </div>
    <img src="02_residual_stream.png" alt="Residual Stream">
    
    <h2>3. Mean Pooling</h2>
    <div class="description">
        Mean pooling averages activations across all tokens to get a fixed-size vector,
        allowing us to compare questions of different lengths.
    </div>
    <img src="03_mean_pooling.png" alt="Mean Pooling">
    
    <h2>4. Linear Probe & Faithfulness Direction</h2>
    <div class="description">
        A linear probe is a simple classifier that learns to predict faithfulness from activations.
        The learned weight vector represents the "faithfulness direction" in activation space.
    </div>
    <img src="04_linear_probe.png" alt="Linear Probe">
    
    <h2>5. Faithful vs Unfaithful Chain-of-Thought</h2>
    <div class="description">
        Faithful: The reasoning genuinely led to the answer.<br>
        Unfaithful: The answer was decided first, then reasoning was generated to justify it.
    </div>
    <img src="05_faithful_vs_unfaithful.png" alt="Faithful vs Unfaithful">
    
    <h2>6. Layer Progression</h2>
    <div class="description">
        Shows how the separation between faithful and unfaithful responses changes across layers.
        Middle layers (12, 18) typically show the best separation.
    </div>
    <img src="06_layer_progression.png" alt="Layer Progression">
    
    <h2>7. Full Pipeline</h2>
    <div class="description">
        The complete flow from question → model response → activations → probe → prediction.
    </div>
    <img src="07_full_pipeline.png" alt="Full Pipeline">
</body>
</html>
"""
    with open(f"{OUTPUT_DIR}/index.html", 'w') as f:
        f.write(html)
    print(f"✓ Saved: {OUTPUT_DIR}/index.html")


def main():
    print("=" * 60)
    print("Generating Technical Concept Visualizations")
    print("=" * 60)
    print()
    
    visualize_activations()
    visualize_residual_stream()
    visualize_mean_pooling()
    visualize_linear_probe()
    visualize_faithfulness_concept()
    visualize_layer_progression()
    visualize_full_pipeline()
    create_index_html()
    
    print()
    print("=" * 60)
    print("✅ All visualizations generated!")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"🌐 Open {OUTPUT_DIR}/index.html in a browser to view all")
    print("=" * 60)


if __name__ == "__main__":
    main()
