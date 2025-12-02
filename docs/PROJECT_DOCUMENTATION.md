# KnoAD-Net: Knowledge-Augmented Alzheimer's Disease Detection Network

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Scientific Background](#scientific-background)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Components](#model-components)
- [Evaluation & Results](#evaluation--results)
- [Technical Details](#technical-details)
- [Future Work](#future-work)
- [References](#references)

---

## 🎯 Project Overview

**KnoAD-Net** is an advanced deep learning system for Alzheimer's Disease (AD) detection that combines:

1. **Multi-Modal Learning**: Integrates MRI brain imaging with clinical/tabular features
2. **Cross-Modal Attention**: Learns relationships between imaging and clinical data
3. **Retrieval-Augmented Generation (RAG)**: Provides explainable, evidence-based diagnostic reasoning
4. **Vision Transformer (ViT)**: Processes MRI scans with state-of-the-art image encoding

### Key Features

✅ **Three-class Classification**: Cognitive Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)  
✅ **Explainable AI**: RAG module provides clinical justifications based on established medical guidelines  
✅ **Multi-Modal Fusion**: Combines visual and tabular data for comprehensive assessment  
✅ **Optimized for CPU**: Configured for AMD Ryzen 5000 processors (also supports GPU)  
✅ **Clinical Decision Support**: Generates diagnostic reports with recommendations

---

## 🧠 Scientific Background

### Alzheimer's Disease

Alzheimer's Disease is a progressive neurodegenerative disorder and the most common cause of dementia, affecting over 55 million people worldwide. Early and accurate diagnosis is critical for:

- Initiating timely interventions
- Planning long-term care
- Enabling participation in clinical trials
- Improving patient and caregiver quality of life

### Diagnostic Challenges

Traditional AD diagnosis relies on:

1. **Clinical Assessment**: MMSE, MoCA cognitive tests (subjective, variable)
2. **Neuroimaging**: MRI, PET scans (requires expert radiologist interpretation)
3. **Biomarkers**: CSF analysis, amyloid/tau PET (expensive, invasive)

**KnoAD-Net addresses these challenges** by:
- Automating MRI analysis with deep learning
- Integrating multiple data modalities
- Providing transparent, guideline-based explanations

### Why Multi-Modal Learning?

Research shows that combining imaging and clinical data significantly improves diagnostic accuracy:

- **MRI imaging** captures structural brain changes (hippocampal atrophy, ventricular enlargement)
- **Clinical features** provide cognitive performance (MMSE scores), demographics, genetic risk (APOE4)
- **Fusion** captures complementary information that single modalities miss

---

## 🏗️ Architecture

### Overall System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         KnoAD-Net System                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   ┌────▼────┐                                 ┌────▼────┐
   │   MRI   │                                 │ Clinical│
   │  Input  │                                 │ Features│
   │(128x128)│                                 │  (9D)   │
   └────┬────┘                                 └────┬────┘
        │                                           │
   ┌────▼────────┐                          ┌───────▼──────┐
   │ Vision      │                          │  Tabular     │
   │ Transformer │                          │  Encoder     │
   │ (ViT-Tiny)  │                          │ (Transformer)│
   │             │                          │              │
   │  768-dim    │                          │   768-dim    │
   └────┬────────┘                          └───────┬──────┘
        │                                           │
        └──────────────┬────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │  Cross-Modal     │
              │  Attention       │
              │  (8 heads)       │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Fusion          │
              │  Classifier      │
              │  (3 classes)     │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │   Prediction     │
              │  CN / MCI / AD   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │   RAG Module     │
              │  (ChromaDB +     │
              │   Clinical KB)   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Diagnostic      │
              │  Report          │
              └──────────────────┘
```

### Key Architectural Decisions

1. **ViT-Tiny Encoder**: Lightweight Vision Transformer optimized for CPU training
2. **Cross-Modal Attention**: Enables bidirectional information flow between modalities
3. **Heavy Regularization**: Dropout (0.3-0.5), label smoothing, L2 weight decay to prevent overfitting
4. **Data Augmentation**: Random rotations, flips, noise to improve generalization
5. **RAG Integration**: Post-hoc explainability using clinical knowledge retrieval

---

## 📊 Dataset

### OASIS-1 (Open Access Series of Imaging Studies)

- **Total Subjects**: ~112 unique subjects (from disc1-disc6)
- **Age Range**: 18-96 years
- **Clinical Data**: Cross-sectional design with detailed demographics

#### MRI Data (Imaging Modality)

- **Format**: T1-weighted MRI scans
- **Original Format**: Analyze format (.hdr/.img) → Converted to NIfTI (.nii.gz)
- **Preprocessing**: 
  - Skull-stripped
  - Normalized to Talairach atlas
  - Bias field corrected (GFC - gradient field correction)
- **Resolution**: 128x128 (downsampled from original)

#### Clinical Features (Tabular Data)

| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Age** | Patient age in years | 18-96 |
| **Sex** | Encoded (0=Female, 1=Male) | 0, 1 |
| **Hand** | Handedness encoded | 0=Right, 1=Left |
| **Educ** | Years of education | 0-23 |
| **SES** | Socioeconomic status (1-5) | 1=Highest, 5=Lowest |
| **MMSE** | Mini-Mental State Exam (0-30) | 24-30=Normal, <24=Impaired |
| **eTIV** | Estimated Total Intracranial Volume | Brain size normalization |
| **nWBV** | Normalized Whole Brain Volume | Atrophy indicator |
| **ASF** | Atlas Scaling Factor | Registration quality |

#### Class Distribution

- **CN (Cognitive Normal)**: ~50-60% of subjects
- **MCI (Mild Cognitive Impairment)**: ~20-30%
- **AD (Alzheimer's Disease)**: ~15-25%

**Note**: Dataset is imbalanced, addressed through:
- Class weighting in loss function
- Stratified train/val/test splits
- Oversampling minority classes during training

---

## 📁 Project Structure

```
Alzimers/
│
├── 📂 data/                          # Dataset (IGNORED in Git - too large)
│   ├── raw/
│   │   ├── mri/
│   │   │   ├── disc1/               # 25 subjects
│   │   │   ├── disc2/               # 22 subjects
│   │   │   ├── disc3/               # 19 subjects
│   │   │   ├── disc4/               # 18 subjects
│   │   │   ├── disc5/               # 20 subjects
│   │   │   └── disc6/               # 8 subjects
│   │   └── tabular/
│   │       └── oasis_cross-sectional.csv
│   ├── processed/                   # Preprocessed MRI slices
│   └── splits/                      # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── 📂 models/                        # Trained model weights
│   └── knoadnet_best.pth
│
├── 📂 checkpoints/                   # Training checkpoints
│
├── 📂 results/                       # Experiment results
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── sample_rag_report.txt
│
├── 📂 visualizations/                # Generated plots
│   ├── attention_maps/
│   ├── data_distribution.png
│   └── training_curves.png
│
├── 📂 logs/                          # Training logs
│
├── 📂 rag_knowledge/                 # RAG knowledge base
│   ├── chroma_db/                   # Vector database
│   └── rag_config.json
│
├── 📂 docs/                          # Documentation (this file!)
│   └── PROJECT_DOCUMENTATION.md
│
├── 📄 config.py                      # Main configuration
├── 📄 utils.py                       # Utility functions
│
├── 📄 phase1_data_pipeline.py       # Data preprocessing
├── 📄 phase2_baselines.py           # Baseline models (CNN, ResNet)
├── 📄 notebook3_knoadnet_core.py    # Main KnoAD-Net model
├── 📄 notebook4_rag_module.py       # RAG implementation
├── 📄 notebook5_evaluation.py       # Comprehensive evaluation
│
├── 📄 convert_hdr_disc1_to_disc6.py # HDR→NIfTI conversion
├── 📄 fix_pipeline_for_disc1to6.py  # Pipeline adapter
├── 📄 RUN_ALL_STEPS.py              # Master execution script
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 .gitignore                     # Git ignore rules
└── 📄 README_DISC1TO6.md            # Quick start guide
```

---

## 🔧 Installation & Setup

### Prerequisites

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **CPU**: Multi-core processor (Ryzen 5000 series recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for dataset and models

### Step 1: Clone Repository

```bash
cd D:\Projects\AI-Projects
# (Repository already exists at Alzimers/)
```

### Step 2: Create Virtual Environment

```powershell
# Windows PowerShell
cd Alzimers
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
cd Alzimers
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download OASIS-1 Dataset

1. Visit: https://www.oasis-brains.org/
2. Register and download OASIS-1 Cross-Sectional dataset
3. Extract disc1 through disc6 to `data/raw/mri/`
4. Place `oasis_cross-sectional.csv` in `data/raw/tabular/`

### Step 5: Verify Installation

```bash
python -c "import torch, nibabel, transformers; print('✓ Installation successful!')"
```

---

## 🚀 Usage Guide

### Quick Start (Automated Pipeline)

The easiest way to run the entire pipeline:

```bash
python RUN_ALL_STEPS.py
```

This interactive script will:
1. ✅ Convert HDR/IMG files to NIfTI format
2. ✅ Preprocess MRI data and create train/val/test splits
3. ✅ Train baseline models (CNN, ResNet)
4. ✅ Train KnoAD-Net model
5. ✅ Generate evaluation reports

### Manual Step-by-Step Execution

#### Step 1: Data Conversion

Convert Analyze format (.hdr/.img) to NIfTI (.nii.gz):

```bash
python convert_hdr_disc1_to_disc6.py
```

**Output**: Converted files in `data/raw/mri/disc*/` folders

#### Step 2: Data Preprocessing

```bash
# Fix pipeline configuration
python fix_pipeline_for_disc1to6.py

# Run preprocessing
python phase1_data_pipeline.py
```

**Output**:
- Processed MRI slices: `data/processed/mri_slices/`
- Train/val/test splits: `data/splits/`

#### Step 3: Train Baseline Models

```bash
python phase2_baselines.py
```

**Output**: CNN and ResNet baseline results in `results/`

#### Step 4: Train KnoAD-Net

```bash
python notebook3_knoadnet_core.py
```

**Output**: 
- Best model: `models/knoadnet_best.pth`
- Training logs: `logs/`
- Attention visualizations: `visualizations/attention_maps/`

#### Step 5: RAG Module

```bash
python notebook4_rag_module.py
```

**Output**:
- Vector database: `rag_knowledge/chroma_db/`
- Sample report: `results/sample_rag_report.txt`

#### Step 6: Comprehensive Evaluation

```bash
python notebook5_evaluation.py
```

**Output**: Complete evaluation metrics, confusion matrices, ROC curves

---

## 🧩 Model Components

### 1. Vision Transformer (ViT) Encoder

- **Architecture**: ViT-Tiny (timm library)
- **Input**: 128×128 grayscale MRI slice
- **Output**: 768-dimensional feature vector
- **Pretrained**: ImageNet (transfer learning)
- **Fine-tuning**: All layers trainable

**Code Snippet**:
```python
self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
# Modify first conv for grayscale input
self.vit.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=16, stride=16)
```

### 2. Tabular Encoder (Transformer)

- **Input**: 9 clinical features
- **Embedding**: Linear projection to 768-dim
- **Architecture**: 3-layer Transformer encoder
- **Output**: 768-dimensional clinical representation

**Code Snippet**:
```python
self.tab_embedding = nn.Linear(n_features, 768)
encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
self.tab_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
```

### 3. Cross-Modal Attention

- **Mechanism**: Multi-head attention between image and tabular features
- **Heads**: 8 attention heads
- **Purpose**: Learn complementary relationships
- **Output**: Fused 768-dim representation

**Mathematical Formulation**:

```
Q = W_q × ImageFeatures
K = W_k × TabularFeatures  
V = W_v × TabularFeatures

Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

### 4. Classification Head

```python
self.classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(256),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(128),
    nn.Linear(128, 3)  # CN, MCI, AD
)
```

### 5. RAG Module

**Components**:

1. **Knowledge Base**: 12 clinical documents covering:
   - NIA-AA diagnostic criteria
   - MMSE/MoCA interpretation
   - MRI biomarkers
   - Treatment guidelines

2. **Embedding**: SentenceTransformer (all-MiniLM-L6-v2)

3. **Vector Store**: ChromaDB for semantic search

4. **Retrieval**: Top-5 most relevant documents for each prediction

5. **Explanation Generator**: Rule-based system (future: LLM integration)

**Workflow**:
```
Model Prediction → Query Construction → Semantic Search → 
Retrieve Clinical Guidelines → Generate Explanation → Diagnostic Report
```

---

## 📈 Evaluation & Results

### Metrics

1. **Accuracy**: Overall classification accuracy
2. **Precision/Recall/F1**: Per-class performance
3. **Confusion Matrix**: Misclassification patterns
4. **ROC-AUC**: One-vs-rest for each class
5. **Attention Analysis**: Heatmaps showing relevant brain regions

### Expected Performance

Based on OASIS-1 dataset (~112 subjects):

| Metric | Baseline CNN | Baseline ResNet | KnoAD-Net |
|--------|--------------|-----------------|-----------|
| Accuracy | ~65-70% | ~70-75% | **~75-80%** |
| F1 (CN) | 0.72 | 0.75 | **0.78** |
| F1 (MCI) | 0.55 | 0.60 | **0.65** |
| F1 (AD) | 0.68 | 0.72 | **0.76** |

**Note**: Small dataset size limits performance. KnoAD-Net's advantage is more pronounced on larger datasets.

### Visualizations

Generated in `visualizations/`:

- **Training curves**: Loss and accuracy over epochs
- **Confusion matrix**: Class-wise predictions
- **ROC curves**: Discriminative ability per class
- **Attention maps**: Highlighting hippocampus, temporal lobes
- **Data distribution**: Class balance, age distribution

---

## ⚙️ Technical Details

### Training Configuration

```python
# From config.py
IMAGE_SIZE = 128             # Reduced for CPU efficiency
BATCH_SIZE = 4               # Small batch for CPU
NUM_EPOCHS = 40              # Moderate for quick training
LEARNING_RATE = 1e-4         # Conservative LR
WEIGHT_DECAY = 0.05          # L2 regularization
WARMUP_EPOCHS = 3            # LR warmup
EARLY_STOPPING_PATIENCE = 10 # Prevent overfitting
```

### Regularization Techniques

1. **Dropout**: 0.3-0.5 throughout network
2. **Label Smoothing**: 0.1 smoothing factor
3. **Weight Decay**: L2 penalty = 0.05
4. **Data Augmentation**: 
   - Random rotations (±15°)
   - Random horizontal flips
   - Gaussian noise (σ=0.01)
5. **Batch Normalization**: Stabilize training

### Loss Function

```python
# CrossEntropyLoss with label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights,      # Address class imbalance
    label_smoothing=0.1        # Prevent overconfidence
)
```

### Optimization

- **Optimizer**: AdamW (Adam with weight decay)
- **Scheduler**: CosineAnnealingLR with warmup
- **Gradient Clipping**: Max norm = 1.0

### Hardware Optimization

**CPU Training** (AMD Ryzen 5000):
```python
torch.set_num_threads(8)     # Match CPU core count
DEVICE = 'cpu'
PIN_MEMORY = False           # No benefit for CPU
```

**Optional GPU Training** (NVIDIA/AMD with ROCm):
```python
if torch.cuda.is_available():
    DEVICE = 'cuda'
    BATCH_SIZE = 16          # Larger batch for GPU
```

---

## 🔮 Future Work

### Short-Term Improvements

1. **Data Augmentation**: 
   - Advanced MRI-specific augmentation
   - Mixup/CutMix for small datasets

2. **Multi-Slice Input**: Use volumetric 3D data instead of single slices

3. **Ensemble Models**: Combine multiple model predictions

4. **Hyperparameter Tuning**: Automated search with Optuna/Ray Tune

### Medium-Term Enhancements

1. **LLM Integration**: Replace rule-based RAG explanations with GPT-4/Gemini

2. **Attention Visualization**: More interpretable saliency maps

3. **Longitudinal Analysis**: Track disease progression over time

4. **External Validation**: Test on ADNI, AIBL datasets

### Long-Term Vision

1. **Multi-Modal Extension**:
   - PET imaging (amyloid, tau, FDG)
   - Genetic data (APOE, polygenic risk scores)
   - CSF biomarkers (Aβ42, p-tau, t-tau)

2. **Clinical Deployment**:
   - Web interface for clinicians
   - DICOM integration
   - HIPAA-compliant data handling
   - Real-time inference API

3. **Federated Learning**: Train across institutions without sharing data

4. **Multimodal Foundation Model**: Pre-train on diverse AD datasets

---

## 📚 References

### Alzheimer's Disease Research

1. **Jack CR Jr et al.** (2011). "Introduction to the recommendations from the National Institute on Aging-Alzheimer's Association." *Alzheimer's & Dementia*, 7(3):257-262.

2. **Jack CR Jr et al.** (2018). "NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease." *Alzheimer's & Dementia*, 14(4):535-562.

3. **Dubois B et al.** (2014). "Advancing research diagnostic criteria for Alzheimer's disease." *Lancet Neurology*, 13(6):614-629.

### OASIS Dataset

4. **Marcus DS et al.** (2007). "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults." *Journal of Cognitive Neuroscience*, 19(9):1498-1507.

### Deep Learning for Medical Imaging

5. **Dosovitskiy A et al.** (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.

6. **Vaswani A et al.** (2017). "Attention is All You Need." *NeurIPS*.

7. **Wen J et al.** (2020). "Convolutional neural networks for classification of Alzheimer's disease." *NeuroImage*, 216:116645.

### Retrieval-Augmented Generation

8. **Lewis P et al.** (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.

9. **Gao L et al.** (2023). "Retrieval-Augmented Multimodal Learning." *CVPR*.

### Medical AI & Explainability

10. **Rudin C** (2019). "Stop explaining black box machine learning models for high stakes decisions." *Nature Machine Intelligence*, 1:206-215.

11. **Ghassemi M et al.** (2021). "The false hope of current approaches to explainable artificial intelligence in health care." *Lancet Digital Health*, 3(11):e745-e750.

---

## 📞 Contact & Support

### Project Information

- **Project Name**: KnoAD-Net (Knowledge-Augmented Alzheimer's Detection Network)
- **Version**: 1.0.0
- **Date Created**: 2025-12-02
- **Primary Dataset**: OASIS-1 Cross-Sectional (~112 subjects, disc1-disc6)

### Troubleshooting

**Common Issues**:

1. **"nibabel not installed"**
   ```bash
   pip install nibabel
   ```

2. **"Out of memory during training"**
   - Reduce `BATCH_SIZE` in `config.py` (try 2 or 1)
   - Reduce `IMAGE_SIZE` to 96 or 64

3. **"ChromaDB persistence error"**
   - Delete `rag_knowledge/chroma_db/` and re-run `notebook4_rag_module.py`

4. **"No MRI files found"**
   - Ensure HDR→NIfTI conversion completed successfully
   - Check that disc1-disc6 folders exist in `data/raw/mri/`

### Getting Help

- Check existing documentation in `docs/`
- Review `README_DISC1TO6.md` for quick start
- Examine log files in `logs/` for error details
- Verify configuration in `config.py`

---

## 📜 License & Ethics

### Data Usage

- **OASIS-1 Dataset**: Publicly available, requires registration at oasis-brains.org
- **Citation Required**: Please cite the OASIS publication (Marcus et al., 2007)

### Research Use Only

⚠️ **IMPORTANT**: This system is for **research purposes only** and is **NOT** approved for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

### Ethical Considerations

- **Bias**: Model performance may vary across demographics (age, sex, education)
- **Privacy**: Ensure HIPAA/GDPR compliance if using patient data
- **Transparency**: Always provide explanations alongside predictions
- **Human Oversight**: Clinical decisions should involve human experts

---

## 🎓 Acknowledgments

This project builds upon:

- **OASIS Team** for providing open-access neuroimaging data
- **timm Library** (Ross Wightman) for Vision Transformer implementations
- **ChromaDB** for vector database infrastructure
- **PyTorch Team** for deep learning framework
- **Medical AI Community** for advancing interpretable healthcare AI

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-02  
**Maintained By**: Project Team

---

*For questions, suggestions, or contributions, please refer to the project repository or contact the maintainers.*
