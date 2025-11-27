# Loss Landscape Analyzer - Quick Start Guide

## 📦 Installatio

### Step 1: Create Project Folder
```bash
mkdir loss_landscape_project
cd loss_landscape_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Save the Files
Save these two files in your project folder:
- `loss_landscape_analysis.py` (the main Python script)
- `requirements.txt` (dependencies list)

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Start the App
```bash
streamlit run loss_landscape_analysis.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 📊 Using the Application

### 1. Configure Your Experiment
In the left sidebar:
- **Architecture**: Choose network depth/structure
  - `shallow`: 2-layer network (baseline)
  - `deep`: 4-layer network (better regularization)
  - `wide`: Wide shallow network
  - `residual`: Skip connections (best performance)

- **Optimizer**: Choose optimization algorithm
  - `SGD`: Finds flatter minima, better generalization
  - `momentum`: Faster convergence with SGD benefits
  - `adam`: Fastest but may find sharper minima

- **Learning Rate**: 0.001 to 0.1 (default: 0.01)
- **Epochs**: 50 to 300 (default: 100)

### 2. Run Analysis
Click the **"🚀 Run Analysis"** button

The app will:
1. Generate synthetic training data
2. Train the neural network
3. Track landscape metrics every 5 epochs
4. Generate visualizations
5. Provide analysis and recommendations

### 3. Interpret Results

#### Training Dynamics (4 plots)
- **Loss Trajectory**: Should decrease smoothly
- **Sharpness**: Lower is better (< 0.1 = excellent)
- **Hessian Max Eigenvalue**: Lower = easier optimization
- **Local Entropy**: Higher = flatter landscape

#### Loss Landscape Visualization
- **3D Surface**: See the full topology
- **Contour Plot**: 2D cross-section with optimizer position

#### Key Metrics
- **Final Loss**: How well the model fits
- **Final Sharpness**: Generalization predictor
- **Max Eigenvalue**: Optimization difficulty
- **Local Entropy**: Flatness indicator

### 4. Export Results
Click **"📥 Download Results CSV"** to get:
- All metrics over training
- Ready for Excel/Python analysis
- Use for creating tables in your report

---

## 🧪 Recommended Experiments

### Experiment 1: Architecture Comparison
```
Run 1: shallow + sgd + lr=0.01 + 100 epochs
Run 2: deep + sgd + lr=0.01 + 100 epochs
Run 3: wide + sgd + lr=0.01 + 100 epochs
Run 4: residual + sgd + lr=0.01 + 100 epochs

Compare: Final sharpness values
Expected: residual < deep < wide < shallow
```

### Experiment 2: Optimizer Effects
```
Run 1: deep + sgd + lr=0.01 + 100 epochs
Run 2: deep + momentum + lr=0.01 + 100 epochs
Run 3: deep + adam + lr=0.01 + 100 epochs

Compare: Final sharpness and convergence speed
Expected: SGD finds flattest minima
```

### Experiment 3: Learning Rate Sensitivity
```
Run 1: deep + sgd + lr=0.005 + 100 epochs
Run 2: deep + sgd + lr=0.01 + 100 epochs
Run 3: deep + sgd + lr=0.05 + 100 epochs

Compare: Convergence stability
Expected: Middle value optimal
```

---

## 📝 For Your Report

### What to Include

1. **Methodology Section**
   - Screenshot of the interface
   - Explain the architecture choices
   - Describe the metrics used

2. **Results Section**
   - Export CSV for all experiments
   - Create comparison tables
   - Include 3D landscape plots
   - Show training dynamics graphs

3. **Analysis Section**
   - Compare architectures quantitatively
   - Discuss optimizer trade-offs
   - Connect findings to theory

4. **Figures to Include**
   - Architecture comparison (bar chart of sharpness)
   - 3D landscape visualizations (4 architectures)
   - Training dynamics (loss + sharpness over time)
   - Correlation plot (sharpness vs generalization)

### Sample Table for Report

| Architecture | Optimizer | Final Loss | Sharpness | Max Eigenvalue | Entropy |
|--------------|-----------|------------|-----------|----------------|---------|
| Shallow      | SGD       | 0.078      | 0.342     | 78.3           | 1.24    |
| Deep         | SGD       | 0.072      | 0.087     | 23.1           | 2.45    |
| Wide         | SGD       | 0.075      | 0.214     | 45.7           | 1.89    |
| Residual     | SGD       | 0.069      | 0.053     | 12.4           | 2.78    |

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Make sure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

### Issue: "torch not found" or CUDA errors
**Solution:** For CPU-only installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Port 8501 already in use"
**Solution:** Use a different port:
```bash
streamlit run loss_landscape_analysis.py --server.port=8502
```

### Issue: Matplotlib displays not showing
**Solution:** Close previous plots:
```bash
# The code already includes plt.close() but if issues persist:
# Restart the Streamlit server
```

### Issue: Takes too long to run
**Solution:** Reduce computational load:
- Decrease epochs: 100 → 50
- Reduce landscape resolution: 30 → 20
- Use shallow architecture for quick tests

---

## 🌐 Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add loss_landscape_analysis.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO
git push -u origin main
```

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `loss_landscape_analysis.py`
6. Click "Deploy"

---

## 📚 Key Papers Referenced

1. **Li et al. (2018)** - Visualizing the Loss Landscape of Neural Nets
   - https://arxiv.org/abs/1712.09913
   - Filter normalization method

2. **Dinh et al. (2017)** - Sharp Minima Can Generalize For Deep Nets
   - https://arxiv.org/abs/1703.04933
   - Reparameterization invariance

3. **Keskar et al. (2016)** - On Large-Batch Training for Deep Learning
   - https://arxiv.org/abs/1609.04836
   - Sharpness-generalization connection


---

## 🎯 Expected Outcomes

After running all experiments, you should observe:

✅ **Deep networks** have lower sharpness than shallow networks  
✅ **Residual connections** produce the flattest landscapes  
✅ **SGD** finds flatter minima than Adam  
✅ **Sharpness** correlates with optimization difficulty  
✅ **Hessian eigenvalues** predict convergence behavior

These findings support the theoretical framework and provide empirical validation for your report.

---

