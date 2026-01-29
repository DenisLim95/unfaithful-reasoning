#!/usr/bin/env python3
"""
Generate figures for MATS application.

Outputs:
- results/figure1_faithfulness_comparison.png
- results/figure2_probe_performance.png  
- results/figure3_accuracy_by_faithfulness.png

Usage:
    python generate_application_figures.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Set style
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)


def load_summaries():
    """Load analysis summaries."""
    
    with open('results/faithfulness_summary.json') as f:
        faith_summary = json.load(f)
    
    with open('results/probe_summary.json') as f:
        probe_summary = json.load(f)
    
    return faith_summary, probe_summary


def figure1_faithfulness_comparison(faith_summary):
    """Figure 1: Faithfulness rate comparison across models."""
    
    print("Generating Figure 1: Faithfulness comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    models = ['Claude 3.7\n(~200B)', 'DeepSeek R1\n(70B)', 'This Work\n(1.5B)']
    faithfulness = [0.25, 0.39, float(faith_summary['faithfulness_rate'].strip('%')) / 100]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # Bar plot
    bars = ax.bar(models, faithfulness, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add percentage labels on bars
    for bar, val in zip(bars, faithfulness):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Faithfulness Rate', fontsize=14, fontweight='bold')
    ax.set_title('CoT Faithfulness: Small vs Large Reasoning Models', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='50% (random)')
    
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/figure1_faithfulness_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results/figure1_faithfulness_comparison.png")


def figure2_probe_performance(faith_summary, probe_summary):
    """Figure 2: Probe performance across layers."""
    
    print("Generating Figure 2: Probe performance...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    results = probe_summary['results_table']
    layers = [r['layer'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    # Plot accuracy
    ax.plot(layers, test_accs, marker='o', linewidth=3, markersize=12, 
            label='Test Accuracy', color='#2ca02c', markeredgecolor='black', markeredgewidth=2)
    
    # Baselines
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2.5, label='Random (50%)', alpha=0.7)
    
    majority = float(faith_summary['faithfulness_rate'].strip('%')) / 100
    ax.axhline(majority, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Majority ({majority:.0%})', alpha=0.7)
    
    # Highlight best layer
    best_layer = probe_summary['best_layer']
    best_acc = float(probe_summary['best_accuracy'].strip('%')) / 100
    ax.scatter([best_layer], [best_acc], s=400, color='gold', 
               edgecolor='black', linewidth=3, zorder=5, label='Best Layer', marker='*')
    
    # Styling
    ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Linear Probe Performance Across Layers', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/figure2_probe_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results/figure2_probe_performance.png")


def figure3_accuracy_by_faithfulness():
    """Figure 3: Q1/Q2 accuracy breakdown by faithfulness."""
    
    print("Generating Figure 3: Accuracy by faithfulness...")
    
    df = pd.read_csv('data/processed/faithfulness_scores.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Q1 accuracy
    q1_faithful = df[df['is_faithful']]['q1_correct'].mean()
    q1_unfaithful = df[~df['is_faithful']]['q1_correct'].mean()
    
    axes[0].bar(['Faithful', 'Unfaithful'], [q1_faithful, q1_unfaithful], 
                color=['#2ca02c', '#d62728'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Q1 Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_title('Q1 Correctness by Faithfulness', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    for i, (val, label) in enumerate(zip([q1_faithful, q1_unfaithful], ['Faithful', 'Unfaithful'])):
        axes[0].text(i, val + 0.03, f'{val:.0%}', ha='center', fontsize=13, fontweight='bold')
    
    # Q2 accuracy
    q2_faithful = df[df['is_faithful']]['q2_correct'].mean()
    q2_unfaithful = df[~df['is_faithful']]['q2_correct'].mean()
    
    axes[1].bar(['Faithful', 'Unfaithful'], [q2_faithful, q2_unfaithful], 
                color=['#2ca02c', '#d62728'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Q2 Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_title('Q2 Correctness by Faithfulness', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    for i, (val, label) in enumerate(zip([q2_faithful, q2_unfaithful], ['Faithful', 'Unfaithful'])):
        axes[1].text(i, val + 0.03, f'{val:.0%}', ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figure3_accuracy_by_faithfulness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results/figure3_accuracy_by_faithfulness.png")


def figure4_layer_comparison_table(probe_summary):
    """Figure 4: Table of probe results (optional, if needed)."""
    
    print("Generating Figure 4: Results table...")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    results = probe_summary['results_table']
    
    table_data = [['Layer', 'Train Acc', 'Test Acc', 'AUC', 'vs Random']]
    for r in results:
        table_data.append([
            f"Layer {r['layer']}",
            f"{r['train_acc']:.1%}",
            f"{r['test_acc']:.1%}",
            f"{r['auc']:.3f}",
            f"{r['vs_random']:+.1f}pp"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row
    best_layer_idx = [r['layer'] for r in results].index(probe_summary['best_layer']) + 1
    for i in range(5):
        table[(best_layer_idx, i)].set_facecolor('#FFD700')
        table[(best_layer_idx, i)].set_text_props(weight='bold')
    
    plt.title('Probe Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('results/figure4_probe_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results/figure4_probe_table.png")


def main():
    """Generate all figures."""
    
    print("=" * 60)
    print("GENERATING APPLICATION FIGURES")
    print("=" * 60)
    print()
    
    # Check that summaries exist
    if not Path('results/faithfulness_summary.json').exists():
        print("✗ Error: Run analyze_final_results.py first")
        return
    
    if not Path('results/probe_summary.json').exists():
        print("✗ Error: Run analyze_final_results.py first")
        return
    
    # Load summaries
    faith_summary, probe_summary = load_summaries()
    
    # Generate figures
    figure1_faithfulness_comparison(faith_summary)
    figure2_probe_performance(faith_summary, probe_summary)
    figure3_accuracy_by_faithfulness()
    figure4_layer_comparison_table(probe_summary)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FIGURES SUMMARY")
    print("=" * 60)
    print("\nGenerated 4 figures in results/:")
    print("  1. figure1_faithfulness_comparison.png")
    print("     → Use in Executive Summary, Finding 1")
    print("  2. figure2_probe_performance.png")
    print("     → Use in Executive Summary, Finding 2")
    print("  3. figure3_accuracy_by_faithfulness.png")
    print("     → Use in Discussion section")
    print("  4. figure4_probe_table.png")
    print("     → Use in Appendix or slides")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run: python extract_examples.py")
    print("2. Review figures and verify they look good")
    print("3. Reference figures in write-up:")
    print("   - ![Figure 1](results/figure1_faithfulness_comparison.png)")
    print("   - ![Figure 2](results/figure2_probe_performance.png)")
    print("4. Fill in remaining [PENDING] sections")
    print("=" * 60)


if __name__ == '__main__':
    main()

