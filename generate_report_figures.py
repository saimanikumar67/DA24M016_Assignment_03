"""
Generate High-Quality Figures for Academic Report
Creates publication-ready visualizations from experiment results

Usage:
python generate_report_figures.py

Requirements:
- Run batch_experiments.py first to generate results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_results():
    """Load experimental results"""
    results_dir = Path('results')
    
    # Load summary
    df = pd.read_csv(results_dir / 'experiment_summary.csv')
    
    # Load detailed results
    with open(results_dir / 'detailed_results.json', 'r') as f:
        detailed = json.load(f)
    
    return df, detailed

def figure1_architecture_comparison(df, output_dir):
    """
    Figure 1: Architecture Impact on Loss Landscape Geometry
    4-panel figure showing key metrics across architectures
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compute means and std across trials
    grouped = df.groupby('Architecture').agg({
        'Final_Sharpness': ['mean', 'std'],
        'Final_Hessian_Max': ['mean', 'std'],
        'Final_Local_Entropy': ['mean', 'std'],
        'Final_Loss': ['mean', 'std']
    })
    
    architectures = ['shallow', 'deep', 'wide', 'residual']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # Panel A: Sharpness
    ax = axes[0, 0]
    means = [grouped.loc[arch, ('Final_Sharpness', 'mean')] for arch in architectures]
    stds = [grouped.loc[arch, ('Final_Sharpness', 'std')] for arch in architectures]
    
    bars = ax.bar(architectures, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('Sharpness', fontweight='bold')
    ax.set_title('(A) Sharpness Metric', fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Excellent threshold')
    ax.legend()
    
    # Panel B: Hessian Max Eigenvalue
    ax = axes[0, 1]
    means = [grouped.loc[arch, ('Final_Hessian_Max', 'mean')] for arch in architectures]
    stds = [grouped.loc[arch, ('Final_Hessian_Max', 'std')] for arch in architectures]
    
    bars = ax.bar(architectures, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('Max Eigenvalue', fontweight='bold')
    ax.set_title('(B) Hessian Curvature', fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel C: Local Entropy
    ax = axes[1, 0]
    means = [grouped.loc[arch, ('Final_Local_Entropy', 'mean')] for arch in architectures]
    stds = [grouped.loc[arch, ('Final_Local_Entropy', 'std')] for arch in architectures]
    
    bars = ax.bar(architectures, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('Local Entropy', fontweight='bold')
    ax.set_title('(C) Landscape Flatness', fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlabel('Architecture', fontweight='bold')
    
    # Panel D: Final Loss
    ax = axes[1, 1]
    means = [grouped.loc[arch, ('Final_Loss', 'mean')] for arch in architectures]
    stds = [grouped.loc[arch, ('Final_Loss', 'std')] for arch in architectures]
    
    bars = ax.bar(architectures, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('Final Loss', fontweight='bold')
    ax.set_title('(D) Training Performance', fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlabel('Architecture', fontweight='bold')
    
    plt.suptitle('Figure 1: Architecture Impact on Loss Landscape Geometry', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_architecture_comparison.pdf')
    plt.savefig(output_dir / 'figure1_architecture_comparison.png', dpi=300)
    print("✓ Generated Figure 1: Architecture Comparison")
    plt.close()

def figure2_optimizer_effects(df, output_dir):
    """
    Figure 2: Optimizer Effects on Landscape Navigation
    Comparing SGD, Momentum, and Adam
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    architectures = ['shallow', 'deep', 'wide', 'residual']
    optimizers = ['sgd', 'momentum', 'adam']
    
    # Panel A: Sharpness
    ax = axes[0]
    pivot = df.groupby(['Architecture', 'Optimizer'])['Final_Sharpness'].mean().unstack()
    pivot = pivot.reindex(architectures)
    
    x = np.arange(len(architectures))
    width = 0.25
    
    for i, opt in enumerate(optimizers):
        ax.bar(x + i*width, pivot[opt], width, label=opt.upper(), alpha=0.8)
    
    ax.set_xlabel('Architecture', fontweight='bold')
    ax.set_ylabel('Sharpness', fontweight='bold')
    ax.set_title('(A) Sharpness by Optimizer', fontweight='bold', loc='left')
    ax.set_xticks(x + width)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel B: Hessian
    ax = axes[1]
    pivot = df.groupby(['Architecture', 'Optimizer'])['Final_Hessian_Max'].mean().unstack()
    pivot = pivot.reindex(architectures)
    
    for i, opt in enumerate(optimizers):
        ax.bar(x + i*width, pivot[opt], width, label=opt.upper(), alpha=0.8)
    
    ax.set_xlabel('Architecture', fontweight='bold')
    ax.set_ylabel('Max Eigenvalue', fontweight='bold')
    ax.set_title('(B) Curvature by Optimizer', fontweight='bold', loc='left')
    ax.set_xticks(x + width)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel C: Loss
    ax = axes[2]
    pivot = df.groupby(['Architecture', 'Optimizer'])['Final_Loss'].mean().unstack()
    pivot = pivot.reindex(architectures)
    
    for i, opt in enumerate(optimizers):
        ax.bar(x + i*width, pivot[opt], width, label=opt.upper(), alpha=0.8)
    
    ax.set_xlabel('Architecture', fontweight='bold')
    ax.set_ylabel('Final Loss', fontweight='bold')
    ax.set_title('(C) Convergence by Optimizer', fontweight='bold', loc='left')
    ax.set_xticks(x + width)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Figure 2: Optimizer Effects on Landscape Navigation', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_optimizer_effects.pdf')
    plt.savefig(output_dir / 'figure2_optimizer_effects.png', dpi=300)
    print("✓ Generated Figure 2: Optimizer Effects")
    plt.close()

def figure3_sharpness_correlation(df, output_dir):
    """
    Figure 3: Sharpness-Loss Correlation Analysis
    Demonstrates theoretical prediction
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Scatter plot
    ax = axes[0]
    
    arch_colors = {'shallow': '#e74c3c', 'deep': '#3498db', 
                   'wide': '#2ecc71', 'residual': '#9b59b6'}
    
    for arch in df['Architecture'].unique():
        subset = df[df['Architecture'] == arch]
        ax.scatter(subset['Final_Sharpness'], subset['Final_Loss'], 
                  label=arch, color=arch_colors[arch], s=100, alpha=0.7, edgecolors='black')
    
    # Add trend line
    z = np.polyfit(df['Final_Sharpness'], df['Final_Loss'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Final_Sharpness'].min(), df['Final_Sharpness'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.5, label='Trend')
    
    # Correlation coefficient
    corr = df['Final_Sharpness'].corr(df['Final_Loss'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Sharpness', fontweight='bold')
    ax.set_ylabel('Final Loss', fontweight='bold')
    ax.set_title('(A) Sharpness-Loss Correlation', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Heatmap
    ax = axes[1]
    
    # Compute correlation matrix
    metrics = ['Final_Loss', 'Final_Sharpness', 'Final_Hessian_Max', 'Final_Local_Entropy']
    corr_matrix = df[metrics].corr()
    
    # Rename for clarity
    corr_matrix.index = ['Loss', 'Sharpness', 'Hessian', 'Entropy']
    corr_matrix.columns = ['Loss', 'Sharpness', 'Hessian', 'Entropy']
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax, fmt='.3f', cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1, linewidths=1, linecolor='black')
    ax.set_title('(B) Metric Correlation Matrix', fontweight='bold', loc='left')
    
    plt.suptitle('Figure 3: Sharpness-Loss Correlation Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_correlation_analysis.pdf')
    plt.savefig(output_dir / 'figure3_correlation_analysis.png', dpi=300)
    print("✓ Generated Figure 3: Correlation Analysis")
    plt.close()

def figure4_training_dynamics(detailed, output_dir):
    """
    Figure 4: Training Dynamics Over Time
    Shows evolution of metrics during training
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select one example from each architecture
    architectures = ['shallow', 'deep', 'wide', 'residual']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # Find SGD examples
    examples = {}
    for result in detailed:
        if result['optimizer'] == 'sgd' and result['trial'] == 0:
            examples[result['architecture']] = result
    
    # Panel A: Loss
    ax = axes[0, 0]
    for arch, color in zip(architectures, colors):
        if arch in examples:
            history = examples[arch]['history']
            epochs = np.arange(len(history['loss'])) * 5
            ax.plot(epochs, history['loss'], label=arch, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('(A) Loss Trajectory', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Sharpness
    ax = axes[0, 1]
    for arch, color in zip(architectures, colors):
        if arch in examples:
            history = examples[arch]['history']
            epochs = np.arange(len(history['sharpness'])) * 5
            ax.plot(epochs, history['sharpness'], label=arch, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Sharpness', fontweight='bold')
    ax.set_title('(B) Sharpness Evolution', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Panel C: Hessian Max
    ax = axes[1, 0]
    for arch, color in zip(architectures, colors):
        if arch in examples:
            history = examples[arch]['history']
            epochs = np.arange(len(history['hessian_max'])) * 5
            ax.plot(epochs, history['hessian_max'], label=arch, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Max Eigenvalue', fontweight='bold')
    ax.set_title('(C) Curvature Evolution', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Panel D: Local Entropy
    ax = axes[1, 1]
    for arch, color in zip(architectures, colors):
        if arch in examples:
            history = examples[arch]['history']
            epochs = np.arange(len(history['local_entropy'])) * 5
            ax.plot(epochs, history['local_entropy'], label=arch, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Local Entropy', fontweight='bold')
    ax.set_title('(D) Flatness Evolution', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Figure 4: Training Dynamics Over Time (SGD)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_training_dynamics.pdf')
    plt.savefig(output_dir / 'figure4_training_dynamics.png', dpi=300)
    print("✓ Generated Figure 4: Training Dynamics")
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate LaTeX table for report"""
    
    # Compute statistics
    stats = df.groupby(['Architecture', 'Optimizer']).agg({
        'Final_Loss': ['mean', 'std'],
        'Final_Sharpness': ['mean', 'std'],
        'Final_Hessian_Max': ['mean', 'std'],
        'Final_Local_Entropy': ['mean', 'std']
    }).round(4)
    
    # Save as CSV
    stats.to_csv(output_dir / 'table1_summary_statistics.csv')
    
    # Generate LaTeX
    latex = stats.to_latex(
        caption="Summary Statistics: Mean (Standard Deviation) across 3 trials",
        label="tab:summary_stats",
        column_format='ll|rr|rr|rr|rr'
    )
    
    with open(output_dir / 'table1_summary_statistics.tex', 'w') as f:
        f.write(latex)
    
    print("✓ Generated Table 1: Summary Statistics (CSV + LaTeX)")

def main():
    """Generate all report figures"""
    
    print("=" * 80)
    print("GENERATING REPORT FIGURES")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print("\nLoading results...")
    df, detailed = load_results()
    
    # Generate figures
    print("\nGenerating figures...")
    figure1_architecture_comparison(df, output_dir)
    figure2_optimizer_effects(df, output_dir)
    figure3_sharpness_correlation(df, output_dir)
    figure4_training_dynamics(detailed, output_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_summary_table(df, output_dir)
    
    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED!")
    print("=" * 80)
    print(f"\nCheck the '{output_dir}' folder for:")
    print("  - figure1_architecture_comparison.png/.pdf")
    print("  - figure2_optimizer_effects.png/.pdf")
    print("  - figure3_correlation_analysis.png/.pdf")
    print("  - figure4_training_dynamics.png/.pdf")
    print("  - table1_summary_statistics.csv/.tex")

if __name__ == "__main__":
    main()