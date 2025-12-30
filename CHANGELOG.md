# Changelog

All notable changes to the KnoAD-Net project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned Features
- 3D volumetric MRI processing support
- LLM-powered RAG explanations (GPT-4/Gemini integration)
- Web-based inference API and dashboard
- Multi-dataset support (ADNI, AIBL)
- Longitudinal disease progression tracking
- Enhanced attention visualization tools
- Automated hyperparameter tuning with Optuna
- Federated learning capabilities

---

## [1.0.0] - 2025-12-30

### Added - Initial Release

#### Core Features
- **Multi-Modal Architecture**: Vision Transformer (ViT) + Tabular Transformer fusion
- **Cross-Modal Attention**: 8-head attention mechanism for imaging-clinical data fusion
- **RAG Module**: Retrieval-Augmented Generation for explainable predictions using ChromaDB
- **Three-Class Classification**: CN (Cognitive Normal), MCI (Mild Cognitive Impairment), AD (Alzheimer's Disease)
- **CPU-Optimized Training**: Configured for AMD Ryzen 5000 series processors
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1, precision, recall, ROC-AUC

#### Data Pipeline
- **HDR to NIfTI Conversion**: Automated conversion script for OASIS-1 dataset (disc1-disc6)
- **Data Preprocessing**: MRI normalization, resizing to 128x128, skull-stripping
- **Clinical Feature Engineering**: Processing of 9 clinical features (Age, Sex, MMSE, Education, etc.)
- **Train/Val/Test Splits**: Stratified splitting with ~70/15/15 ratio
- **Data Augmentation**: Random rotations, flips, brightness adjustments, noise injection

#### Model Components
- **Vision Encoder**: ViT-Tiny (timm library) pretrained on ImageNet
- **Tabular Encoder**: 3-layer Transformer encoder for clinical features
- **Cross-Modal Fusion**: Multi-head attention with layer normalization
- **Classification Head**: Deep MLP with dropout (0.5) and batch normalization
- **Regularization**: L2 weight decay (0.05), label smoothing (0.1), heavy dropout

#### Training & Optimization
- **Loss Function**: CrossEntropyLoss with class weights and label smoothing
- **Optimizer**: AdamW with learning rate 1e-4
- **Scheduler**: CosineAnnealingLR with 3-epoch warmup
- **Early Stopping**: Patience of 10 epochs
- **Gradient Clipping**: Max norm of 1.0

#### Baseline Models
- **Simple CNN**: 4-layer convolutional baseline
- **ResNet18**: Transfer learning baseline
- **Tabular-Only**: MLP classifier on clinical features only
- **ViT Baseline**: ViT without multi-modal fusion

#### RAG & Explainability
- **Knowledge Base**: 12 clinical documents on AD diagnosis, MMSE interpretation, MRI biomarkers
- **Vector Database**: ChromaDB with sentence-transformers embeddings (all-MiniLM-L6-v2)
- **Semantic Search**: Top-5 relevant documents for each prediction
- **Diagnostic Reports**: Evidence-based explanations with clinical guidelines

#### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score per class
- **Confusion Matrix**: Visualization of classification patterns
- **ROC Curves**: One-vs-rest curves for each class
- **Attention Maps**: Heatmaps highlighting important brain regions
- **Training Curves**: Loss and accuracy plots over epochs
- **Data Distribution**: Visualization of class balance and demographics

#### Documentation
- **README.md**: Comprehensive project overview and quick start guide
- **PROJECT_DOCUMENTATION.md**: Complete technical documentation (50+ pages)
- **README_DISC1TO6.md**: Quick start guide for OASIS-1 disc1-disc6 processing
- **WORKFLOW.py**: Step-by-step workflow instructions
- **CONTRIBUTING.md**: Contribution guidelines and coding standards
- **CODE_OF_CONDUCT.md**: Community guidelines and ethics
- **LICENSE**: MIT License with medical research use terms
- **CHANGELOG.md**: Version history (this file)

#### Scripts & Utilities
- **RUN_ALL_STEPS.py**: Interactive master script for complete pipeline execution
- **convert_hdr_disc1_to_disc6.py**: Automated HDR/IMG to NIfTI conversion
- **fix_pipeline_for_disc1to6.py**: Pipeline adapter for disc1-disc6 data
- **phase1_data_pipeline.py**: Complete data preprocessing pipeline
- **phase2_baselines.py**: Baseline model training and evaluation
- **notebook3_knoadnet_core.py**: Main KnoAD-Net model implementation
- **notebook4_rag_module.py**: RAG module setup and knowledge base creation
- **notebook5_evaluation.py**: Comprehensive model evaluation
- **config.py**: Centralized configuration management
- **utils.py**: Common utility functions

#### Dataset Support
- **OASIS-1**: Open Access Series of Imaging Studies (Cross-Sectional)
- **Subjects**: ~112 unique subjects from disc1-disc6
- **Age Range**: 18-96 years
- **MRI Format**: T1-weighted, skull-stripped, Talairach normalized
- **Clinical Features**: Age, Sex, Handedness, Education, SES, MMSE, eTIV, nWBV, ASF

#### Dependencies
- PyTorch 2.0+
- torchvision 0.15+
- timm 0.9+ (Vision Transformers)
- transformers 4.30+ (HuggingFace)
- nibabel 5.0+ (Medical imaging)
- ChromaDB 0.4+ (Vector database)
- sentence-transformers 2.2+ (Embeddings)
- scikit-learn 1.3+ (Metrics)
- pandas 2.0+ (Data processing)
- matplotlib/seaborn (Visualization)

### Performance
- **Training**: ~40 epochs, ~30-45 minutes on AMD Ryzen 5000 series
- **Inference**: ~50ms per sample on CPU
- **Accuracy**: ~75-80% on OASIS-1 test set
- **F1 Scores**: CN: 0.78, MCI: 0.65, AD: 0.76

### Known Issues
- Small dataset size (~112 subjects) limits maximum achievable performance
- Class imbalance (more CN samples than MCI/AD) addressed with weighted loss
- Single-slice 2D processing - full 3D volumetric support planned
- RAG explanations are rule-based - LLM integration planned
- CPU training is slower than GPU (but more accessible)

### Security
- No known security vulnerabilities
- All medical data handling follows best practices
- No patient-identifiable information in code or documentation

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-12-30 | Initial release with full pipeline, multi-modal model, and RAG |

---

## Migration Guide

### From Development to v1.0.0

This is the first official release. If you were using development versions:

1. **Update Dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Configuration Changes**:
   - Review `config.py` for any local path updates
   - Ensure `PROJECT_ROOT` points to your installation directory

3. **Data Pipeline**:
   - Use `RUN_ALL_STEPS.py` for automated setup
   - Or follow manual steps in `README_DISC1TO6.md`

4. **Model Checkpoints**:
   - Old checkpoints may not be compatible
   - Retrain models using updated scripts

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting issues
- Suggesting features
- Submitting pull requests
- Code style and testing

---

## Citation

If you use KnoAD-Net in your research, please cite:

```bibtex
@software{knoadnet2025,
  title={KnoAD-Net: Knowledge-Augmented Alzheimer's Detection Network},
  author={KnoAD-Net Contributors},
  year={2025},
  url={https://github.com/Aspect022/Alzimers-Research},
  version={1.0.0}
}
```

And please cite the OASIS dataset:

```bibtex
@article{marcus2007oasis,
  title={Open access series of imaging studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults},
  author={Marcus, Daniel S and Wang, Tracy H and Parker, Jamie and Csernansky, John G and Morris, John C and Buckner, Randy L},
  journal={Journal of cognitive neuroscience},
  volume={19},
  number={9},
  pages={1498--1507},
  year={2007}
}
```

---

## Acknowledgments

- **OASIS Team** for providing open-access neuroimaging data
- **Ross Wightman** for timm library (Vision Transformers)
- **HuggingFace** for transformers library
- **PyTorch Team** for the deep learning framework
- **ChromaDB** for vector database infrastructure
- All contributors to this project

---

## Links

- **Repository**: https://github.com/Aspect022/Alzimers-Research
- **Documentation**: [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)
- **Issues**: https://github.com/Aspect022/Alzimers-Research/issues
- **OASIS Dataset**: https://www.oasis-brains.org/

---

**Note**: This is a research project. See [LICENSE](LICENSE) for important terms regarding medical use and liability.
