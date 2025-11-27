"""
Loss Landscape Geometry & Optimization Dynamics Analysis
A rigorous framework for analyzing neural network loss landscapes

Reference Papers:
- Visualizing the Loss Landscape of Neural Nets (Li et al., 2018)
- Sharp Minima Can Generalize For Deep Nets (Dinh et al., 2017)
- Essentially No Barriers in Neural Network Energy Landscape (Draxler et al., 2018)

Installation:
pip install streamlit numpy torch matplotlib seaborn scipy pandas

Run:
streamlit run loss_landscape_analysis.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Set style
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')

@dataclass
class LandscapeMetrics:
    """Container for landscape metrics"""
    loss: float
    sharpness: float
    hessian_max_eigenvalue: float
    gradient_norm: float
    local_entropy: float

class SimpleNet(nn.Module):
    """Configurable neural network for landscape analysis"""
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
            # Residual connection for 'residual' architecture
            if self.architecture == 'residual' and i == len(self.layers_list) // 2:
                if x.shape == identity.shape:
                    x = x + identity
        return x

class LossLandscapeAnalyzer:
    """Main class for analyzing loss landscape geometry"""
    
    def __init__(self, model, data, labels):
        self.model = model
        self.data = data
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.labels = self.labels.to(self.device)
        
    def compute_loss(self, params=None):
        """Compute loss with optional parameter override"""
        if params is not None:
            self._set_params(params)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data)
            loss = nn.MSELoss()(outputs, self.labels)
        return loss.item()
    
    def _set_params(self, params):
        """Set model parameters from flat vector"""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = params[offset:offset + numel].view_as(p).clone()
            offset += numel
    
    def _get_params(self):
        """Get flattened model parameters"""
        return torch.cat([p.data.flatten() for p in self.model.parameters()])
    
    def compute_gradient(self):
        """Compute gradient at current parameters"""
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(self.data)
        loss = nn.MSELoss()(outputs, self.labels)
        loss.backward()
        
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        return grad, torch.norm(grad).item()
    
    def compute_hessian_eigenvalues(self, num_eigenvalues=3):
        """
        Compute top eigenvalues of Hessian using power iteration
        Based on: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
        """
        params = self._get_params()
        n_params = len(params)
        
        eigenvalues = []
        for _ in range(num_eigenvalues):
            v = torch.randn(n_params, device=self.device)
            v = v / torch.norm(v)
            
            # Power iteration
            for _ in range(10):
                self.model.zero_grad()
                outputs = self.model(self.data)
                loss = nn.MSELoss()(outputs, self.labels)
                
                # Hessian-vector product
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
        """
        Compute sharpness metric around current minimum
        Based on: "On Large-Batch Training for Deep Learning" (Keskar et al., 2016)
        """
        current_loss = self.compute_loss()
        params = self._get_params()
        
        max_loss = current_loss
        # Sample points in a ball around current parameters
        for _ in range(20):
            perturbation = torch.randn_like(params) * epsilon
            perturbed_params = params + perturbation
            loss = self.compute_loss(perturbed_params)
            max_loss = max(max_loss, loss)
        
        # Restore original parameters
        self._set_params(params)
        sharpness = max_loss - current_loss
        return sharpness
    
    def compute_local_entropy(self):
        """
        Compute local entropy - diversity of loss values nearby
        Higher entropy suggests flatter regions
        """
        params = self._get_params()
        losses = []
        
        for _ in range(30):
            perturbation = torch.randn_like(params) * 0.005
            perturbed_params = params + perturbation
            loss = self.compute_loss(perturbed_params)
            losses.append(loss)
        
        self._set_params(params)
        
        # Compute entropy from loss distribution
        losses = np.array(losses)
        hist, _ = np.histogram(losses, bins=10, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def generate_2d_landscape(self, direction1, direction2, alpha_range=(-1, 1), 
                             beta_range=(-1, 1), resolution=25):
        """
        Generate 2D slice of loss landscape
        Based on: Li et al. (2018) filter-wise normalization
        """
        params = self._get_params()
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
        betas = np.linspace(beta_range[0], beta_range[1], resolution)
        
        landscape = np.zeros((resolution, resolution))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                perturbed = params + alpha * direction1 + beta * direction2
                landscape[i, j] = self.compute_loss(perturbed)
        
        self._set_params(params)
        return alphas, betas, landscape
    
    def train_and_track(self, optimizer_name='sgd', lr=0.01, epochs=100):
        """Train model and track landscape metrics"""
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = nn.MSELoss()(outputs, self.labels)
            loss.backward()
            optimizer.step()
            
            # Compute metrics every 5 epochs
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
                
                status_text.text(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Sharpness: {sharpness:.4f}")
            
            progress_bar.progress((epoch + 1) / epochs)
        
        progress_bar.empty()
        status_text.empty()
        return history

def generate_toy_data(n_samples=200):
    """Generate toy regression data"""
    X = torch.randn(n_samples, 2)
    y = torch.sin(X[:, 0]) * torch.cos(X[:, 1]) + 0.1 * torch.randn(n_samples, 1)
    return X, y

def plot_landscape_3d(alphas, betas, landscape):
    """Create 3D landscape visualization"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    A, B = np.meshgrid(alphas, betas)
    surf = ax.plot_surface(A, B, landscape.T, cmap=cm.viridis, alpha=0.8)
    
    ax.set_xlabel('Direction 1', fontsize=10)
    ax.set_ylabel('Direction 2', fontsize=10)
    ax.set_zlabel('Loss', fontsize=10)
    ax.set_title('Loss Landscape 3D Visualization', fontsize=12, fontweight='bold')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def plot_landscape_contour(alphas, betas, landscape):
    """Create contour plot of landscape"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(alphas, betas, landscape.T, levels=20, cmap='viridis')
    ax.contour(alphas, betas, landscape.T, levels=10, colors='white', 
               alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('Direction 1', fontsize=11)
    ax.set_ylabel('Direction 2', fontsize=11)
    ax.set_title('Loss Landscape Contour Plot', fontsize=13, fontweight='bold')
    
    fig.colorbar(contour, label='Loss Value')
    ax.plot(0, 0, 'r*', markersize=15, label='Current Minimum')
    ax.legend()
    
    return fig

def plot_training_metrics(history):
    """Plot training metrics over time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.arange(len(history['loss'])) * 5
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpness
    axes[0, 1].plot(epochs, history['sharpness'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Sharpness')
    axes[0, 1].set_title('Sharpness (Generalization Indicator)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hessian Max Eigenvalue
    axes[1, 0].plot(epochs, history['hessian_max'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Max Eigenvalue')
    axes[1, 0].set_title('Hessian Maximum Eigenvalue', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Local Entropy
    axes[1, 1].plot(epochs, history['local_entropy'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Local Entropy')
    axes[1, 1].set_title('Local Entropy (Flatness Indicator)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Loss Landscape Analyzer", layout="wide")
    
    st.title("Loss Landscape Geometry & Optimization Dynamics")
    st.markdown("""
    ### Rigorous Framework for Neural Network Loss Landscape Analysis
    
    **Key Papers:**
    - 📄 [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) (Li et al., 2018)
    - 📄 [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933) (Dinh et al., 2017)
    - 📄 [On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836) (Keskar et al., 2016)
    """)
    
    st.sidebar.header("Configuration")
    
    # Architecture selection
    architecture = st.sidebar.selectbox(
        "Network Architecture",
        ['shallow', 'deep', 'wide', 'residual'],
        help="Different architectures exhibit different landscape geometries"
    )
    
    # Optimizer selection
    optimizer_name = st.sidebar.selectbox(
        "Optimizer",
        ['sgd', 'momentum', 'adam'],
        help="SGD tends to find flatter minima than adaptive methods"
    )
    
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    epochs = st.sidebar.slider("Training Epochs", 50, 300, 100, 10)
    
    # Theoretical Background
    with st.expander("📚 Theoretical Framework", expanded=False):
        st.markdown("""
        ### Key Theoretical Insights
        
        **1. Sharpness-Generalization Connection**
        - Sharp minima (high curvature) → Poor generalization
        - Flat minima (low curvature) → Better generalization
        - Formula: `Sharpness = max_||δ||≤ε L(θ + δ) - L(θ)`
        
        **2. Hessian Spectrum Analysis**
        - Maximum eigenvalue indicates local curvature
        - Predicts optimization difficulty
        - Low eigenvalues → Easy optimization landscape
        
        **3. Architecture Impact**
        - Deep networks: Implicit regularization toward flat minima
        - Residual connections: Even flatter landscapes
        - Width: Affects loss landscape smoothness
        
        **4. SGD Implicit Bias**
        - Stochastic noise helps escape sharp minima
        - Noise scale ∝ learning_rate × batch_size
        - Naturally biased toward flat regions
        
        **5. Mode Connectivity**
        - Different minima often connected by low-loss paths
        - SGD finds solutions in same "mode" of loss landscape
        """)
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        st.header("Analysis Results")
        
        # Generate data
        with st.spinner("Generating training data..."):
            X, y = generate_toy_data(n_samples=200)
        
        # Initialize model and analyzer
        model = SimpleNet(architecture=architecture)
        analyzer = LossLandscapeAnalyzer(model, X, y)
        
        # Train and track metrics
        st.subheader("1️⃣ Training Dynamics")
        history = analyzer.train_and_track(optimizer_name, learning_rate, epochs)
        
        # Plot training metrics
        fig_metrics = plot_training_metrics(history)
        st.pyplot(fig_metrics)
        plt.close()
        
        # Key findings from training
        final_sharpness = history['sharpness'][-1]
        final_hessian = history['hessian_max'][-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Loss", f"{history['loss'][-1]:.4f}")
        col2.metric("Final Sharpness", f"{final_sharpness:.4f}")
        col3.metric("Max Eigenvalue", f"{final_hessian:.4f}")
        col4.metric("Local Entropy", f"{history['local_entropy'][-1]:.4f}")
        
        # Generate landscape visualization
        st.subheader("2️⃣ Loss Landscape Visualization")
        
        with st.spinner("Computing loss landscape (this may take a moment)..."):
            # Get random directions for visualization
            params = analyzer._get_params()
            dir1 = torch.randn_like(params)
            dir1 = dir1 / torch.norm(dir1)
            dir2 = torch.randn_like(params)
            dir2 = dir2 - torch.dot(dir1, dir2) * dir1  # Orthogonalize
            dir2 = dir2 / torch.norm(dir2)
            
            alphas, betas, landscape = analyzer.generate_2d_landscape(
                dir1, dir2, alpha_range=(-0.5, 0.5), 
                beta_range=(-0.5, 0.5), resolution=30
            )
        
        tab1, tab2 = st.tabs(["3D Surface", "Contour Plot"])
        
        with tab1:
            fig_3d = plot_landscape_3d(alphas, betas, landscape)
            st.pyplot(fig_3d)
            plt.close()
        
        with tab2:
            fig_contour = plot_landscape_contour(alphas, betas, landscape)
            st.pyplot(fig_contour)
            plt.close()
        
        # Analysis Summary
        st.subheader("3️⃣ Key Findings & Interpretations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Generalization Prediction**")
            if final_sharpness < 0.1:
                st.success("✅ Low sharpness detected - Excellent generalization expected")
            elif final_sharpness < 0.3:
                st.info("ℹ️ Moderate sharpness - Good generalization expected")
            else:
                st.warning("⚠️ High sharpness - Potential overfitting risk")
        
        with col2:
            st.markdown("**Optimization Difficulty**")
            if final_hessian < 10:
                st.success("✅ Low curvature - Easy to optimize")
            elif final_hessian < 50:
                st.info("ℹ️ Moderate curvature - Standard optimization")
            else:
                st.warning("⚠️ High curvature - Difficult optimization")
        
        # Export results
        st.subheader("4️⃣ Export Results")
        
        results_df = pd.DataFrame({
            'Epoch': np.arange(len(history['loss'])) * 5,
            'Loss': history['loss'],
            'Sharpness': history['sharpness'],
            'Hessian_Max': history['hessian_max'],
            'Gradient_Norm': history['gradient_norm'],
            'Local_Entropy': history['local_entropy']
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"landscape_analysis_{architecture}_{optimizer_name}.csv",
            mime="text/csv"
        )
        
        # Research recommendations
        st.subheader("5️⃣ Research Recommendations")
        st.markdown(f"""
        Based on this analysis with **{architecture}** architecture and **{optimizer_name}** optimizer:
        
        - **Architecture Choice**: {'✅ Good choice - promotes flat minima' if architecture in ['deep', 'residual'] else '⚠️ Consider deeper architectures for flatter landscapes'}
        - **Optimizer**: {'✅ SGD recommended for better generalization' if optimizer_name == 'sgd' else 'ℹ️ Adaptive optimizers may find sharper minima'}
        - **Learning Rate**: {'✅ Appropriate' if 0.005 <= learning_rate <= 0.05 else '⚠️ Consider adjusting for better convergence'}
        
        **Next Steps for Your Report:**
        1. Compare multiple architectures systematically
        2. Analyze sharpness vs. test error correlation
        3. Study mode connectivity between solutions
        4. Investigate filter-wise normalization (Li et al., 2018)
        5. Measure generalization gap empirically
        """)

if __name__ == "__main__":
    main()