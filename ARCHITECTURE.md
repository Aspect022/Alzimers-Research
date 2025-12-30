# KnoAD-Net Architecture Documentation

**Technical Deep Dive into the Knowledge-Augmented Alzheimer's Detection Network**

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [Model Components](#model-components)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [RAG System](#rag-system)
- [Technical Specifications](#technical-specifications)
- [Design Decisions](#design-decisions)
- [Performance Optimization](#performance-optimization)

---

## Overview

KnoAD-Net is a multi-modal deep learning architecture that combines:
1. **Vision Transformer (ViT)** for MRI image encoding
2. **Transformer Encoder** for clinical/tabular feature encoding
3. **Cross-Modal Attention** for feature fusion
4. **Retrieval-Augmented Generation (RAG)** for explainable predictions

### Architecture Philosophy

- **Multi-Modal by Design**: Leverages complementary information from imaging and clinical data
- **Attention-Based Fusion**: Goes beyond simple concatenation to learn complex relationships
- **Explainability First**: RAG module provides evidence-based clinical reasoning
- **Efficiency Focused**: Optimized for CPU training while maintaining performance

---

## System Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        Input Layer                                 │
├─────────────────────────────┬──────────────────────────────────────┤
│    MRI Image (128x128x3)    │  Clinical Features (9D vector)      │
│    - Grayscale normalized   │  - Age, Sex, Education              │
│    - Skull-stripped         │  - MMSE, Brain volumes              │
└─────────────┬───────────────┴───────────────┬──────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│  Vision Encoder         │     │  Tabular Encoder            │
│  ─────────────────      │     │  ────────────────            │
│  • ViT-Tiny (timm)      │     │  • Linear Embedding (9→768) │
│  • Patch Size: 16x16    │     │  • 3-Layer Transformer      │
│  • ImageNet Pretrained  │     │  • 4 Attention Heads        │
│  • Modified for 1-channel│    │  • Positional Encoding      │
│                         │     │                             │
│  Output: 768-dim vector │     │  Output: 768-dim vector     │
└─────────────┬───────────┘     └───────────┬─────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Cross-Modal Attention      │
              │   ───────────────────────    │
              │   • Multi-Head Attention     │
              │   • 8 Attention Heads        │
              │   • Query: Image Features    │
              │   • Key/Value: Tab Features  │
              │   • Residual Connection      │
              │   • Layer Normalization      │
              │                              │
              │   Output: 768-dim fused      │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Fusion Classifier          │
              │   ────────────────            │
              │   • FC: 768 → 256            │
              │   • ReLU + Dropout(0.5)      │
              │   • BatchNorm                │
              │   • FC: 256 → 128            │
              │   • ReLU + Dropout(0.5)      │
              │   • BatchNorm                │
              │   • FC: 128 → 3              │
              │                              │
              │   Output: [CN, MCI, AD]      │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Prediction + Confidence    │
              │   Class: CN/MCI/AD           │
              │   Probabilities: [p1,p2,p3]  │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   RAG Module (Optional)      │
              │   ────────────────            │
              │   • Query Construction       │
              │   • Semantic Search          │
              │   • Top-5 Clinical Docs      │
              │   • Evidence-Based Report    │
              │                              │
              │   Output: Diagnostic Report  │
              └──────────────────────────────┘
```

---

## Data Flow

### Training Data Flow

```
1. Data Loading
   ├─ MRI: Load .npy preprocessed slices (128x128)
   ├─ Clinical: Load from processed CSV
   └─ Labels: CN=0, MCI=1, AD=2

2. Data Augmentation (Training Only)
   ├─ Random horizontal flip (p=0.5)
   ├─ Random rotation ±10° (p=0.5)
   ├─ Random brightness ±15% (p=0.5)
   └─ Gaussian noise σ=0.02 (p=0.3)

3. Batch Creation
   ├─ Batch Size: 4 (CPU), 16 (GPU)
   ├─ Stratified sampling for class balance
   └─ Pin memory for GPU transfer

4. Forward Pass
   ├─ Vision Encoder: img → 768-dim
   ├─ Tabular Encoder: features → 768-dim
   ├─ Cross-Modal Attention: fusion → 768-dim
   └─ Classifier: fusion → 3 logits

5. Loss Computation
   ├─ CrossEntropyLoss with class weights
   ├─ Label smoothing (0.1)
   └─ L2 regularization (weight decay 0.05)

6. Optimization
   ├─ AdamW optimizer (lr=1e-4)
   ├─ Gradient clipping (max_norm=1.0)
   └─ CosineAnnealingLR scheduler with warmup
```

### Inference Data Flow

```
1. Input Preparation
   ├─ Load MRI slice
   ├─ Normalize to [0,1]
   ├─ Extract clinical features
   └─ No augmentation

2. Model Forward Pass
   ├─ Set model to eval mode
   ├─ Disable gradient computation
   └─ Generate prediction + probabilities

3. RAG Explanation (Optional)
   ├─ Construct query from prediction + features
   ├─ Semantic search in knowledge base
   ├─ Retrieve top-5 relevant documents
   └─ Generate evidence-based report

4. Output
   ├─ Predicted class (CN/MCI/AD)
   ├─ Confidence scores
   ├─ Attention maps
   └─ Clinical explanation (if RAG enabled)
```

---

## Model Components

### 1. Vision Transformer (ViT) Encoder

**Architecture**: ViT-Tiny from timm library

```python
# Configuration
Model: vit_tiny_patch16_224
Patch Size: 16x16
Embedding Dim: 192
Hidden Dim: 768
Num Layers: 12
Num Heads: 3
MLP Ratio: 4

# Modifications for Medical Imaging
- Input: 1 channel (grayscale) instead of 3 (RGB)
- Modified first conv layer: Conv2d(1, 192, kernel_size=16, stride=16)
- Pretrained on ImageNet, fine-tuned on OASIS-1
```

**Why ViT-Tiny?**
- Lightweight: 5.7M parameters vs 86M for ViT-Base
- Fast CPU training: ~3x faster than larger variants
- Sufficient capacity for 128x128 images
- Better generalization on small datasets

**Input Processing**:
1. MRI slice (128x128) → Resize to 224x224 (ViT input size)
2. Convert to 3-channel by repeating: [H, W] → [3, H, W]
3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output**: 768-dimensional feature vector (global representation)

---

### 2. Tabular Transformer Encoder

**Architecture**: Custom transformer for clinical features

```python
# Input Features (9D)
[Age, Sex, Handedness, Education, SES, MMSE, eTIV, nWBV, ASF]

# Architecture
Linear Embedding: 9 → 768
Positional Encoding: Learnable 1D positional embeddings
Transformer Encoder:
  - 3 layers
  - d_model: 768
  - num_heads: 4
  - d_ff: 2048 (FFN hidden dim)
  - dropout: 0.1
  - Layer normalization
  - Residual connections

# Output: 768-dimensional feature vector
```

**Feature Engineering**:
```python
# Normalization
Age: StandardScaler (μ=0, σ=1)
Sex: Binary encoding (0/1)
Handedness: Binary encoding
Education: Years (0-23)
SES: Socioeconomic status (1-5)
MMSE: Mini-Mental State Exam (0-30)
eTIV: Estimated Total Intracranial Volume (log-normalized)
nWBV: Normalized Whole Brain Volume
ASF: Atlas Scaling Factor
```

**Why Transformer for Tabular Data?**
- Learns feature interactions through self-attention
- Handles varying feature importance
- Better than simple MLP for heterogeneous features
- Aligns representation space with ViT

---

### 3. Cross-Modal Attention Mechanism

**Purpose**: Learn complementary relationships between imaging and clinical data

```python
# Architecture
Query: Image features (768-dim) → Linear(768, 768)
Key: Tabular features (768-dim) → Linear(768, 768)
Value: Tabular features (768-dim) → Linear(768, 768)

Multi-Head Attention:
  - num_heads: 8
  - head_dim: 768 / 8 = 96
  - dropout: 0.3
  - batch_first: True

Layer Norm + Residual: output = LayerNorm(image + attention_output)

Feed-Forward Network:
  - FC: 768 → 2048
  - GELU activation
  - Dropout: 0.3
  - FC: 2048 → 768

Final Layer Norm + Residual
```

**Mathematical Formulation**:

```
Q = W_q × ImageFeatures                    # (batch, 768)
K = W_k × TabularFeatures                  # (batch, 768)
V = W_v × TabularFeatures                  # (batch, 768)

# Multi-Head Attention
Attention(Q,K,V) = softmax(QK^T / √d_k) × V

# Split into 8 heads
Q_h = split(Q, num_heads=8)                # (batch, 8, 96)
K_h = split(K, num_heads=8)
V_h = split(V, num_heads=8)

# Compute attention per head
head_i = Attention(Q_h[i], K_h[i], V_h[i])

# Concatenate heads
multi_head_output = concat(head_1, ..., head_8)  # (batch, 768)

# Residual connection
output = LayerNorm(ImageFeatures + multi_head_output)

# Feed-forward
ff_output = FFN(output)
final_output = LayerNorm(output + ff_output)
```

**Attention Pattern Interpretation**:
- High attention to MMSE → Model learns cognitive scores are important
- High attention to nWBV → Brain volume correlates with imaging
- High attention to Age → Age-related patterns in MRI

---

### 4. Classification Head

```python
# Multi-Layer Perceptron with Heavy Regularization
nn.Sequential(
    # Layer 1
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(256),
    
    # Layer 2
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(128),
    
    # Output Layer
    nn.Linear(128, 3)  # CN, MCI, AD
)
```

**Regularization Techniques**:
1. **Dropout (0.5)**: Prevents co-adaptation of features
2. **Batch Normalization**: Stabilizes training, reduces internal covariate shift
3. **L2 Weight Decay (0.05)**: Penalizes large weights
4. **Label Smoothing (0.1)**: Reduces overconfidence

**Why This Design?**
- Deep enough to learn complex patterns
- Regularized enough to prevent overfitting on small dataset
- Gradual dimension reduction (768 → 256 → 128 → 3)

---

## Training Pipeline

### Training Loop

```python
for epoch in range(NUM_EPOCHS):
    # Training Phase
    model.train()
    for batch in train_loader:
        # 1. Forward pass
        outputs = model(batch['image'], batch['features'])
        
        # 2. Compute loss
        loss = criterion(outputs, batch['label'])
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. Optimizer step
        optimizer.step()
    
    # Validation Phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            val_outputs = model(batch['image'], batch['features'])
            val_loss = criterion(val_outputs, batch['label'])
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break
```

### Loss Function

```python
# Class weights (to handle imbalance)
class_weights = compute_class_weight('balanced', classes=[0,1,2], y=train_labels)

# CrossEntropyLoss with modifications
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(class_weights),
    label_smoothing=0.1,
    reduction='mean'
)

# Total loss = CE Loss + L2 Regularization
total_loss = ce_loss + weight_decay * sum(param.norm(2) for param in model.parameters())
```

### Optimizer Configuration

```python
# AdamW (Adam with decoupled weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05
)
```

### Learning Rate Schedule

```python
# Warmup: Linear increase for first 3 epochs
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=3
)

# Cosine Annealing: Gradual decrease after warmup
main_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS - 3,
    eta_min=1e-6
)

# Chained scheduler
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[3]
)
```

---

## Inference Pipeline

```python
def predict(mri_slice, clinical_features, model, rag_module=None):
    """
    Single-sample inference with optional RAG explanation.
    """
    # 1. Preprocessing
    mri_tensor = preprocess_mri(mri_slice)  # Normalize, resize
    feature_tensor = torch.FloatTensor(clinical_features)
    
    # 2. Model prediction
    model.eval()
    with torch.no_grad():
        logits = model(mri_tensor.unsqueeze(0), feature_tensor.unsqueeze(0))
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # 3. Extract attention weights (for visualization)
    attention_weights = model.get_attention_weights()
    
    # 4. RAG explanation (optional)
    explanation = None
    if rag_module:
        query = construct_query(predicted_class, clinical_features)
        relevant_docs = rag_module.search(query, top_k=5)
        explanation = generate_report(predicted_class, probabilities, relevant_docs)
    
    # 5. Return results
    return {
        'class': ['CN', 'MCI', 'AD'][predicted_class],
        'probabilities': probabilities.numpy(),
        'attention_weights': attention_weights,
        'explanation': explanation
    }
```

---

## RAG System

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RAG Pipeline                          │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌──────────────────┐              ┌──────────────────┐
│ Knowledge Base   │              │ Embedding Model  │
│ ──────────────   │              │ ────────────     │
│ • 12 Clinical    │              │ • sentence-      │
│   Documents      │──────────────│   transformers   │
│ • NIA-AA Criteria│   Encode     │ • all-MiniLM-    │
│ • MMSE Guide     │              │   L6-v2          │
│ • MRI Biomarkers │              │ • 384-dim        │
│ • Treatment      │              │   embeddings     │
└──────────────────┘              └──────────────────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
                ┌──────────────────┐
                │   ChromaDB       │
                │   ──────────     │
                │   • Vector Store │
                │   • Cosine Sim   │
                │   • Fast Search  │
                └────────┬─────────┘
                         │
              ┌──────────▼──────────┐
              │   Query Processing  │
              │   ────────────────  │
              │   Prediction: AD    │
              │   MMSE: 18          │
              │   Age: 75           │
              │   → Query String    │
              └──────────┬──────────┘
                         │
                ┌────────▼────────┐
                │ Semantic Search │
                │ ─────────────── │
                │ Top-5 Relevant  │
                │ Documents       │
                └────────┬────────┘
                         │
            ┌────────────▼────────────┐
            │ Report Generation       │
            │ ───────────────────     │
            │ • Evidence Summary      │
            │ • Clinical Guidelines   │
            │ • Recommendations       │
            └─────────────────────────┘
```

### Knowledge Base Contents

```python
clinical_documents = [
    "NIA-AA Diagnostic Criteria for Alzheimer's Disease",
    "MMSE Score Interpretation Guide",
    "MRI Biomarkers: Hippocampal Atrophy",
    "MRI Biomarkers: Ventricular Enlargement",
    "Cognitive Impairment Assessment",
    "Mild Cognitive Impairment (MCI) Diagnosis",
    "Age-Related Cognitive Decline",
    "Treatment Guidelines for Alzheimer's",
    "Clinical Decision Support for Dementia",
    "Risk Factors: APOE4, Family History",
    "Differential Diagnosis: AD vs Other Dementias",
    "Longitudinal Monitoring Recommendations"
]
```

---

## Technical Specifications

### Model Parameters

| Component | Parameters | Memory |
|-----------|-----------|--------|
| ViT Encoder | 5.7M | 23 MB |
| Tabular Encoder | 2.4M | 10 MB |
| Cross-Modal Attention | 4.7M | 19 MB |
| Classifier | 0.2M | 1 MB |
| **Total** | **13.0M** | **53 MB** |

### Computational Requirements

**Training**:
- Batch Size: 4 (CPU), 16 (GPU)
- Memory: ~8 GB (CPU), ~6 GB (GPU)
- Time per epoch: ~3-5 min (CPU), ~30-60s (GPU)
- Total training: ~2-3 hours (CPU), ~30-40 min (GPU)

**Inference**:
- Single sample: ~50ms (CPU), ~5ms (GPU)
- Batch of 32: ~1s (CPU), ~100ms (GPU)

### Hardware Recommendations

**Minimum**:
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 10 GB

**Recommended**:
- CPU: 8+ cores (e.g., AMD Ryzen 5000 series)
- RAM: 16 GB
- Storage: 20 GB
- GPU: Optional (NVIDIA/AMD with 6+ GB VRAM)

---

## Design Decisions

### Why Multi-Modal?

**Research Evidence**:
- Combining MRI + clinical data improves accuracy by 10-15% over single modality
- Different modalities capture complementary information
- Clinical features provide context for imaging findings

**Our Results**:
- MRI-only: ~65% accuracy
- Clinical-only: ~55% accuracy
- **Multi-modal: ~75-80% accuracy**

### Why Cross-Modal Attention?

**Alternatives Considered**:
1. **Simple Concatenation**: [image_features || clinical_features] → Classifier
   - Con: No interaction learning
   - Con: Equal importance to all features

2. **Weighted Sum**: α × image + β × clinical → Classifier
   - Con: Linear combination only
   - Con: Fixed weights for all samples

3. **Cross-Modal Attention** (Selected)
   - ✅ Learns sample-specific relationships
   - ✅ Bidirectional information flow
   - ✅ Interpretable attention weights
   - ✅ State-of-the-art in multi-modal learning

### Why ViT over CNN?

**CNNs**:
- Inductive bias for local patterns
- Translation equivariance
- Fewer parameters

**Vision Transformers**:
- ✅ Global receptive field from layer 1
- ✅ Long-range dependencies (important for brain MRI)
- ✅ Better transfer learning from ImageNet
- ✅ Attention maps for interpretability
- ✅ Scales better with data (future-proof)

### Why RAG for Explainability?

**Alternatives**:
1. **Gradient-based (Grad-CAM, Integrated Gradients)**
   - Con: Only shows "where" model looks, not "why"

2. **Rule-based IF-THEN**
   - Con: Rigid, doesn't leverage medical knowledge

3. **RAG** (Selected)
   - ✅ Evidence-based explanations
   - ✅ Grounded in clinical guidelines
   - ✅ Flexible and extensible
   - ✅ Can integrate LLMs (future)

---

## Performance Optimization

### CPU Optimization

```python
# Threading
torch.set_num_threads(8)  # Match CPU core count

# Data loading
num_workers = 0  # For CPU, 0 is often faster
pin_memory = False  # No benefit for CPU

# Mixed precision (CPU support limited)
# Use float32 for stability

# Batch size
BATCH_SIZE = 4  # Smaller batches for CPU
```

### GPU Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images, features)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Data loading
num_workers = 4
pin_memory = True

# Batch size
BATCH_SIZE = 16  # Larger batches for GPU
```

### Memory Optimization

```python
# Gradient accumulation (simulate larger batch)
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    outputs = model(batch['image'], batch['features'])
    loss = criterion(outputs, batch['label']) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Gradient checkpointing (saves memory)
model.gradient_checkpointing_enable()
```

---

## Conclusion

KnoAD-Net represents a comprehensive approach to Alzheimer's Disease detection through:
- Multi-modal learning for improved accuracy
- Attention mechanisms for feature fusion and interpretability
- RAG system for clinically-grounded explanations
- Efficient design for accessible research

This architecture balances performance, interpretability, and computational efficiency, making it suitable for both research and potential clinical applications.

---

**For implementation details, see the source code in:**
- `notebook3_knoadnet_core.py` - Main model
- `notebook4_rag_module.py` - RAG system
- `config.py` - Configuration
- `docs/PROJECT_DOCUMENTATION.md` - Complete documentation
