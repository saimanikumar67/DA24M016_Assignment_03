"""
Batch Experiment Runner for Loss Landscape Analysis
Runs systematic experiments for comprehensive report

Usage:
python batch_experiments.py

This will:
1. Run all architecture-optimizer combinations
2. Save results to CSV files
3. Generate comparison plots
4. Create statistical analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Import from the main file (make sure loss_landscape_analysis.py is in same directory)
import sys
sys.path.append('.')

class SimpleNet(nn.Module):
    """Configurable neural network"""
    def __init__(self, architecture='shallow'):
        super().__init__()
        architectures = {
            'shallow': [2, 20, 1],
            'deep': [2, 16, 16, 16, 1],
            'wide': [2, 100, 1],
            'residual': [2, 20, 20, 1]
        }
        
        layers = architectures[architecture]
        self.architecture = architecture
        self.layers_list = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers_list.append(nn.ReLU())
    
    def forward(self, x):
        identity = x
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            if self.architecture == 'residual' and i == len(self.layers_list) // 2:
                if x.shape == identity.shape:
                    x = x + identity
        return x

class LossLandscapeAnalyzer:
    """Loss landscape analyzer"""
    def __init__(self, model, data, labels):
        self.model = model
        self.data = data
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.labels = self.labels.to(self.device)
        
    def compute_loss(self, params=None):
        if params is not None:
            self._set_params(params)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data)
            loss = nn.MSELoss()(outputs, self.labels)
        return loss.item()
    
    def _set_params(self, params):
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = params[offset:offset + numel].view_as(p).clone()
            offset += numel
    
    def _get_params(self):
        return torch.cat([p.data.flatten() for p in self.model.parameters()])
    
    def compute_gradient(self):
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(self.data)
        loss = nn.MSELoss()(outputs, self.labels)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        return grad, torch.norm(grad).item()
    
    def compute_hessian_eigenvalues(self, num_eigenvalues=2):
        params = self._get_params()
        n_params = len(params)
        eigenvalues = []
        
        for _ in range(num_eigenvalues):
            v = torch.randn(n_params, device=self.device)
            v = v / torch.norm(v)
            
            for _ in range(10):
                self.model.zero_grad()
                outputs = self.model(self.data)
                loss = nn.MSELoss()(outputs, self.labels)
                grads = torch.autograd.grad(loss, self.model.parameters(), 
                                           create_graph=True, retain_graph=True)
                grad_vec = torch.cat([g.flatten() for g in grads])
                hvp = torch.autograd.grad(grad_vec, self.model.parameters(),
                                         grad_outputs=v, retain_graph=True)
                hvp = torch.cat([g.flatten() for g in hvp if g is not None])
                v = hvp / (torch.norm(hvp) + 1e-8)
            
            eigenvalue = torch.dot(v, hvp).item()
            eigenvalues.append(eigenvalue)
        
        return max(eigenvalues), eigenvalues
    
    def compute_sharpness(self, epsilon=0.01):
        current_loss = self.compute_loss()
        params = self._get_params()
        max_loss = current_loss
        
        for _ in range(20):
            perturbation = torch.randn_like(params) * epsilon
            perturbed_params = params + perturbation
            loss = self.compute_loss(perturbed_params)
            max_loss = max(max_loss, loss)
        
        self._set_params(params)
        return max_loss - current_loss
    
    def compute_local_entropy(self):
        params = self._get_params()
        losses = []
        
        for _ in range(30):
            perturbation = torch.randn_like(params) * 0.005
            perturbed_params = params + perturbation
            loss = self.compute_loss(perturbed_params)
            losses.append(loss)
        
        self._set_params(params)
        losses = np.array(losses)
        hist, _ = np.histogram(losses, bins=10, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def train_and_track(self, optimizer_name='sgd', lr=0.01, epochs=100):
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_name == 'momentum':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        history = {
            'loss': [], 'sharpness': [], 'hessian_max': [],
            'gradient_norm': [], 'local_entropy': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = nn.MSELoss()(outputs, self.labels)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                _, grad_norm = self.compute_gradient()
                sharpness = self.compute_sharpness()
                hessian_max, _ = self.compute_hessian_eigenvalues(num_eigenvalues=2)
                entropy = self.compute_local_entropy()
                
                history['loss'].append(loss.item())
                history['sharpness'].append(sharpness)
                history['hessian_max'].append(hessian_max)
                history['gradient_norm'].append(grad_norm)
                history['local_entropy'].append(entropy)
        
        return history

def generate_toy_data(n_samples=200, seed=42):
    """Generate reproducible toy data"""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, 2)
    y = torch.sin(X[:, 0]) * torch.cos(X[:, 1]) + 0.1 * torch.randn(n_samples, 1)
    return X, y

def run_single_experiment(architecture, optimizer, lr, epochs, trial, data):
    """Run a single experiment"""
    X, y = data
    model = SimpleNet(architecture=architecture)
    analyzer = LossLandscapeAnalyzer(model, X, y)
    
    history = analyzer.train_and_track(optimizer, lr, epochs)
    
    return {
        'architecture': architecture,
        'optimizer': optimizer,
        'learning_rate': lr,
        'epochs': epochs,
        'trial': trial,
        'final_loss': history['loss'][-1],
        'final_sharpness': history['sharpness'][-1],
        'final_hessian_max': history['hessian_max'][-1],
        'final_gradient_norm': history['gradient_norm'][-1],
        'final_local_entropy': history['local_entropy'][-1],
        'history': history
    }

def run_batch_experiments():
    """Run all experiments systematically"""
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Experiment configurations
    architectures = ['shallow', 'deep', 'wide', 'residual']
    optimizers = ['sgd', 'momentum', 'adam']
    learning_rates = [0.01]
    epochs = 100
    n_trials = 3  # Multiple trials for statistical significance
    
    # Generate data once
    data = generate_toy_data(n_samples=200, seed=42)
    
    # Run experiments
    all_results = []
    total_experiments = len(architectures) * len(optimizers) * len(learning_rates) * n_trials
    
    print(f"Running {total_experiments} experiments...")
    print("=" * 80)
    
    with tqdm(total=total_experiments) as pbar:
        for arch in architectures:
            for opt in optimizers:
                for lr in learning_rates:
                    for trial in range(n_trials):
                        # Run experiment
                        result = run_single_experiment(arch, opt, lr, epochs, trial, data)
                        all_results.append(result)
                        
                        # Update progress
                        pbar.set_description(f"{arch}-{opt} (trial {trial+1}/{n_trials})")
                        pbar.update(1)
                        
                        # Print summary
                        print(f"\n{arch:10} + {opt:10} | Loss: {result['final_loss']:.4f} | "
                              f"Sharpness: {result['final_sharpness']:.4f} | "
                              f"Hessian: {result['final_hessian_max']:.4f}")
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    
    # Save results
    save_results(all_results, output_dir)
    
    # Generate analysis
    analyze_results(all_results, output_dir)
    
    return all_results

def save_results(results, output_dir):
    """Save results to CSV and JSON"""
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'Architecture': r['architecture'],
            'Optimizer': r['optimizer'],
            'Trial': r['trial'],
            'Final_Loss': r['final_loss'],
            'Final_Sharpness': r['final_sharpness'],
            'Final_Hessian_Max': r['final_hessian_max'],
            'Final_Gradient_Norm': r['final_gradient_norm'],
            'Final_Local_Entropy': r['final_local_entropy']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'experiment_summary.csv', index=False)
    print(f"\n✓ Saved summary to: {output_dir / 'experiment_summary.csv'}")
    
    # Save detailed results as JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed results to: {output_dir / 'detailed_results.json'}")

def analyze_results(results, output_dir):
    """Generate statistical analysis and plots"""
    
    # Create DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'Architecture': r['architecture'],
            'Optimizer': r['optimizer'],
            'Trial': r['trial'],
            'Loss': r['final_loss'],
            'Sharpness': r['final_sharpness'],
            'Hessian': r['final_hessian_max'],
            'Entropy': r['final_local_entropy']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Compute statistics
    stats = df.groupby(['Architecture', 'Optimizer']).agg({
        'Loss': ['mean', 'std'],
        'Sharpness': ['mean', 'std'],
        'Hessian': ['mean', 'std'],
        'Entropy': ['mean', 'std']
    }).round(4)
    
    stats.to_csv(output_dir / 'statistics.csv')
    print(f"✓ Saved statistics to: {output_dir / 'statistics.csv'}")
    
    # Generate plots
    generate_comparison_plots(df, output_dir)

def generate_comparison_plots(df, output_dir):
    """Generate comparison plots"""
    
    sns.set_style("whitegrid")
    
    # 1. Architecture Comparison (averaged over optimizers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Sharpness', 'Hessian', 'Entropy', 'Loss']
    titles = ['Sharpness (Lower = Better)', 'Max Hessian Eigenvalue', 
              'Local Entropy (Higher = Better)', 'Final Loss']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Group by architecture and compute mean
        arch_data = df.groupby('Architecture')[metric].mean().sort_values()
        
        bars = ax.bar(range(len(arch_data)), arch_data.values, 
                      color=sns.color_palette("husl", len(arch_data)))
        ax.set_xticks(range(len(arch_data)))
        ax.set_xticklabels(arch_data.index, rotation=45)
        ax.set_ylabel(metric)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'architecture_comparison.png'}")
    plt.close()
    
    # 2. Optimizer Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['Sharpness', 'Loss', 'Entropy']):
        ax = axes[idx]
        
        # Create grouped bar chart
        arch_order = ['shallow', 'deep', 'wide', 'residual']
        opt_order = ['sgd', 'momentum', 'adam']
        
        pivot_data = df.groupby(['Architecture', 'Optimizer'])[metric].mean().unstack()
        pivot_data = pivot_data.reindex(arch_order)
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} by Architecture & Optimizer', fontweight='bold')
        ax.set_xlabel('Architecture')
        ax.set_ylabel(metric)
        ax.legend(title='Optimizer')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(arch_order, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'optimizer_comparison.png'}")
    plt.close()
    
    # 3. Correlation Analysis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute correlation matrix
    corr_data = df[['Loss', 'Sharpness', 'Hessian', 'Entropy']].corr()
    
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, fmt='.3f', cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix of Landscape Metrics', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'correlation_matrix.png'}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the 'results' folder for all outputs.")

def print_summary_table(results):
    """Print formatted summary table"""
    
    df = pd.DataFrame([{
        'Architecture': r['architecture'],
        'Optimizer': r['optimizer'],
        'Loss': r['final_loss'],
        'Sharpness': r['final_sharpness'],
        'Hessian': r['final_hessian_max'],
        'Entropy': r['final_local_entropy']
    } for r in results])
    
    # Group by architecture and optimizer, compute mean
    summary = df.groupby(['Architecture', 'Optimizer']).mean().round(4)
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Mean across trials)")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)

if __name__ == "__main__":
    print("=" * 80)
    print("LOSS LANDSCAPE BATCH EXPERIMENTS")
    print("=" * 80)
    print("\nThis will run systematic experiments for your report.")
    print("Total time: ~15-20 minutes")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")
    
    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nExperiments cancelled.")
        sys.exit(0)
    
    # Run all experiments
    results = run_batch_experiments()
    
    # Print summary
    print_summary_table(results)
    
    print("\nCheck the 'results' folder for:")
    print("  - experiment_summary.csv (all data)")
    print("  - statistics.csv (mean and std)")
    print("  - detailed_results.json (full histories)")
    print("  - architecture_comparison.png (bar charts)")
    print("  - optimizer_comparison.png (grouped bars)")
    print("  - correlation_matrix.png (heatmap)")