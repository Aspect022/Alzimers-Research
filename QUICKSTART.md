# Quick Start Guide

**Get KnoAD-Net running in 5 minutes!** ⚡

---

## ⚡ Super Fast Setup

### 1. Clone & Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Aspect022/Alzimers-Research.git
cd Alzimers-Research

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (Manual - 10 minutes)

1. Visit https://www.oasis-brains.org/
2. Register (free)
3. Download OASIS-1 Cross-Sectional dataset (disc1-disc6)
4. Extract to:
   ```
   Alzimers-Research/data/raw/mri/disc1/
   Alzimers-Research/data/raw/mri/disc2/
   ...
   Alzimers-Research/data/raw/mri/disc6/
   Alzimers-Research/data/raw/tabular/oasis_cross-sectional.csv
   ```

### 3. Run Everything (1 command - 2-3 hours)

```bash
python RUN_ALL_STEPS.py
```

This will:
- ✅ Convert MRI files (HDR → NIfTI)
- ✅ Preprocess data
- ✅ Train baseline models
- ✅ Train KnoAD-Net
- ✅ Setup RAG module
- ✅ Generate evaluation reports

**Done! 🎉** Check `results/` for outputs.

---

## 🎯 What You'll Get

After running the pipeline, you'll have:

```
results/
├── comprehensive_metrics.json     # Model performance metrics
├── confusion_matrix.png           # Classification visualization
├── sample_rag_report.txt         # Example explainable prediction
└── final_evaluation_report.txt   # Complete evaluation

models/
└── knoadnet_best.pth             # Trained model weights

visualizations/
├── training_curves.png           # Loss/accuracy over epochs
├── attention_maps/               # What the model "sees"
└── data_distribution.png         # Dataset statistics
```

---

## 🚀 Manual Step-by-Step (If You Prefer)

### Step 1: Convert MRI Files

```bash
python convert_hdr_disc1_to_disc6.py
```

**Output**: `.nii.gz` files in each disc folder  
**Time**: ~5 minutes

### Step 2: Preprocess Data

```bash
python phase1_data_pipeline.py
```

**Output**: 
- `data/processed/mri_slices/` - Preprocessed images
- `data/splits/` - train/val/test splits  
**Time**: ~15 minutes

### Step 3: Train Baselines

```bash
python phase2_baselines.py
```

**Output**: `results/*_baseline_results.json`  
**Time**: ~30 minutes

### Step 4: Train KnoAD-Net

```bash
python notebook3_knoadnet_core.py
```

**Output**: 
- `models/knoadnet_best.pth` - Best model
- `visualizations/` - Training curves  
**Time**: ~1-2 hours (CPU)

### Step 5: Setup RAG

```bash
python notebook4_rag_module.py
```

**Output**: 
- `rag_knowledge/chroma_db/` - Vector database
- `results/sample_rag_report.txt` - Example report  
**Time**: ~5 minutes

### Step 6: Evaluate

```bash
python notebook5_evaluation.py
```

**Output**: Comprehensive metrics and visualizations  
**Time**: ~10 minutes

---

## 🔧 Quick Configuration

### CPU vs GPU

**Default (CPU)**:
```python
# config.py
DEVICE = 'cpu'
BATCH_SIZE = 4
```

**For GPU** (if you have one):
```python
# config.py
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16  # Larger batch for GPU
```

### Adjust for Your System

**Low RAM (< 16GB)**:
```python
# config.py
BATCH_SIZE = 2
IMAGE_SIZE = 96  # Smaller images
```

**Fast CPU (16+ cores)**:
```python
# config.py
NUM_WORKERS = 8
torch.set_num_threads(16)
```

---

## 📊 Expected Results

After training, you should see:

```
KnoAD-Net Performance on OASIS-1:
=====================================
Accuracy: ~75-80%

Per-Class F1 Scores:
  CN (Cognitive Normal):      0.78
  MCI (Mild Cognitive Impair): 0.65
  AD (Alzheimer's Disease):   0.76

Training Time: ~2-3 hours (CPU)
Inference Time: ~50ms per sample
```

---

## ❓ Troubleshooting

### "nibabel not found"
```bash
pip install nibabel
```

### "Out of memory"
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 2  # or even 1
```

### "No subjects found"
- Check that you extracted disc1-disc6 to `data/raw/mri/`
- Ensure CSV file is in `data/raw/tabular/`
- Run conversion script first: `python convert_hdr_disc1_to_disc6.py`

### "CUDA out of memory"
Switch to CPU:
```python
# config.py
DEVICE = 'cpu'
```

---

## 📚 Next Steps

Once everything is running:

1. **Explore Results**: Check `results/` and `visualizations/`
2. **Read Docs**: See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)
3. **Understand Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Try Inference**: Use trained model on new MRI scans
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎓 Learn More

### Full Documentation
- [README.md](README.md) - Project overview
- [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) - Complete technical docs
- [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into model design

### Key Concepts
- **Multi-Modal Learning**: Combines MRI + clinical data
- **Vision Transformer (ViT)**: Processes brain images
- **Cross-Modal Attention**: Fuses different data types
- **RAG (Retrieval-Augmented Generation)**: Explains predictions

### Research Context
- **Dataset**: OASIS-1 (~112 subjects, ages 18-96)
- **Task**: 3-class classification (CN/MCI/AD)
- **Approach**: Deep learning + explainable AI
- **Goal**: Early Alzheimer's detection

---

## 💡 Tips for Success

1. **Be Patient**: First run takes time (preprocessing + training)
2. **Check Logs**: Look in `logs/` if something fails
3. **Start Small**: Try with fewer epochs first to test
4. **GPU Optional**: CPU works fine, just slower
5. **Ask Questions**: Open an issue if stuck!

---

## ⚡ TL;DR

```bash
# Three commands to rule them all:
git clone https://github.com/Aspect022/Alzimers-Research.git
cd Alzimers-Research
pip install -r requirements.txt

# Download OASIS-1 dataset (manual)
# Then:
python RUN_ALL_STEPS.py

# Wait 2-3 hours, get trained model! 🎉
```

---

## 🆘 Need Help?

- **Quick Questions**: Check [README.md](README.md) or [docs/](docs/)
- **Bugs**: [Report an issue](https://github.com/Aspect022/Alzimers-Research/issues/new?template=bug_report.md)
- **Discussions**: [GitHub Discussions](https://github.com/Aspect022/Alzimers-Research/discussions)
- **Documentation**: [Full Documentation](docs/PROJECT_DOCUMENTATION.md)

---

**Ready to start? Run `python RUN_ALL_STEPS.py` and you're off! 🚀**
