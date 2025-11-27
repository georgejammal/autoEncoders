# Report Generation Summary

## Files Created

### 1. LaTeX Report
- **File**: `REPORT.tex`
- **Format**: Professional LaTeX document
- **Quality**: Publication-ready

### 2. Markdown Report
- **File**: `REPORT.md`
- **Format**: GitHub-flavored Markdown
- **Quality**: Web-ready with embedded images

### 3. Loss Curves
- **Location**: `report_assets/loss_curves/`
- **Format**: High-quality PNG (300 DPI)
- **Count**: 17 plots (one per model)

### 4. Reconstructions
- **Location**: `report_assets/reconstructions/`
- **Models**: Part B2 (4 models), Part C (5 models), Part D3 (1 model)
- **Format**: Side-by-side comparisons (original | reconstruction)

## How to Generate PDF

### Option 1: Overleaf (Recommended)
1. Go to https://www.overleaf.com
2. Create a new project → Upload Project
3. Upload `REPORT.tex` and the `report_assets/` folder
4. Click "Recompile"
5. Download PDF

### Option 2: Local LaTeX Installation
If you have LaTeX installed on your local machine:
```bash
pdflatex REPORT.tex
pdflatex REPORT.tex  # Run twice for TOC
```

### Option 3: Markdown to PDF Converters
Use the `REPORT.md` file with online converters:
- https://www.markdowntopdf.com/
- https://md2pdf.netlify.app/
- https://dillinger.io/ (export as PDF)

### Option 4: Pandoc (if available)
```bash
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex
```

## Report Contents

### For Each of 17 Models:

1. **Model Name** - Clear identification
2. **Architecture Details**:
   - Model class
   - Layer configuration
   - Normalization type
   - Activation functions
   - Special features

3. **Hyperparameters Table**:
   - Epochs
   - Batch size
   - Learning rate
   - Optimizer
   - Loss function
   - Special weights (LPIPS, KLD)

4. **Loss Curves**:
   - High-quality matplotlib plots
   - Training loss (blue with markers)
   - Validation loss (purple with markers)
   - Proper axis labels with numbers
   - Grid for readability

## Models Covered

### Part A (4 models)
- A1: Skip ConvAE (Depth 10)
- A2: Skip ConvAE (BatchNorm)
- A3: Skip ConvAE (LeakyReLU)
- A4: Skip ConvAE (Wide)

### Part B (5 models)
- B1: ResNet AE (Baseline)
- B2.1-B2.4: ResNet + LPIPS (weights: 0.0, 0.05, 0.075, 0.10)

### Part C (5 models)
- C1: VAE (β=0.00025)
- C2: VAE (β=0.001)
- C3: VAE (β=0.01)
- C4: VAE (Wide)
- C5: VAE + LPIPS

### Part D (3 models)
- D1: Deep ResNet VAE
- D2: Deep ResNet VAE + LPIPS
- D3: Deep ResNet AE + Attention

## Next Steps

1. **Review the Markdown report** in your editor to ensure all content is correct
2. **Choose a PDF conversion method** from the options above
3. **Add qualitative results** (reconstruction images) if needed
4. **Add quantitative comparison table** summarizing all models

## File Locations

```
/home/ML_courses/03683533_2025/ameer_george_abdallah/
├── REPORT.tex                          # LaTeX source
├── REPORT.md                           # Markdown version
├── report_assets/
│   ├── loss_curves/                    # 17 PNG plots
│   │   ├── run_A1_skip_convae_depth10.png
│   │   ├── run_A2_skip_convae_bn.png
│   │   └── ... (15 more)
│   └── reconstructions/                # Best reconstructions
│       ├── run_B2_resnet_lpips_w000/
│       ├── run_C1_vae_beta_small/
│       └── ... (10 more)
└── experiments/runs/                   # Original experiment data
```

All assets are ready for your report submission! 📊✨
