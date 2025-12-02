"""
KnoAD-Net Configuration
=======================
Configuration for LOCAL execution on Windows
Optimized for Ryzen 5000 + AMD Radeon (CPU training)
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS - UPDATE THESE TO YOUR LOCAL PATHS!
# ============================================================================

PROJECT_ROOT = Path(r"D:\Projects\AI-Projects\Alzimers")

DIRS = {
    'root': PROJECT_ROOT,
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'data_raw_mri': PROJECT_ROOT / 'data' / 'raw' / 'mri',  # disc1 for half dataset
    'data_raw_tabular': PROJECT_ROOT / 'data' / 'raw' / 'tabular',
    'data_processed': PROJECT_ROOT / 'data' / 'processed',
    'data_splits': PROJECT_ROOT / 'data' / 'splits',
    'models': PROJECT_ROOT / 'models',
    'checkpoints': PROJECT_ROOT / 'checkpoints',
    'results': PROJECT_ROOT / 'results',
    'logs': PROJECT_ROOT / 'logs',
    'visualizations': PROJECT_ROOT / 'visualizations',
    'rag_knowledge': PROJECT_ROOT / 'rag_knowledge',
}

def create_directories():
    """Create all necessary directories"""
    for name, path in DIRS.items():
        path.mkdir(exist_ok=True, parents=True)
    print(f"✓ Created directory structure at: {PROJECT_ROOT}")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET = 'OASIS1'
N_CLASSES = 3
CLASS_NAMES = ['CN', 'MCI', 'AD']

# Feature columns from OASIS-1 CSV
FEATURE_COLS = [
    'Age', 'sex_encoded', 'hand_encoded', 'Educ', 'SES',
    'MMSE', 'eTIV', 'nWBV', 'ASF'
]

# ============================================================================
# MODEL CONFIGURATION (Optimized for CPU)
# ============================================================================

IMAGE_SIZE = 128  # Reduced for faster CPU processing (vs 224)
BATCH_SIZE = 4    # Small batch for CPU (vs 8-16 for GPU)

# Use smaller model variants for CPU
MODEL_CONFIG = {
    'vit_model': 'vit_tiny_patch16_224',  # Tiny instead of Small/Base
    'vit_pretrained': True,
    'vit_hidden_dim': 192,  # Tiny ViT dimension
    'tab_embed_dim': 128,   # Reduced from 256
    'tab_layers': 3,        # Reduced from 4
    'tab_heads': 4,         # Reduced from 8
    'fusion_heads': 4,      # Reduced from 8
}

# ============================================================================
# TRAINING CONFIGURATION (CPU Optimized)
# ============================================================================

# Training parameters
NUM_EPOCHS = 40        # Reasonable for CPU (vs 50-100 for GPU)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3      # Reduced warmup
EARLY_STOPPING_PATIENCE = 10

# Data loading
NUM_WORKERS = 4        # Match your CPU cores (Ryzen usually has 6-8)
PIN_MEMORY = False     # Only useful for CUDA

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

DEVICE = 'cpu'  # Force CPU (AMD GPU not well supported by PyTorch)

# CPU optimization flags
import torch
torch.set_num_threads(8)  # Set to your CPU thread count (Ryzen 5000: 6-12 threads)

# Optional: Try to use AMD GPU if ROCm is installed
# Uncomment this if you've installed PyTorch with ROCm support
# try:
#     if torch.cuda.is_available():  # ROCm shows as CUDA
#         DEVICE = 'cuda'
#         print("✓ AMD GPU detected via ROCm")
# except:
#     pass

print(f"✓ Using device: {DEVICE}")
print(f"✓ CPU threads: {torch.get_num_threads()}")

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# FILE FORMAT SUPPORT
# ============================================================================

# OASIS-1 uses Analyze format (.hdr/.img), not NIfTI
MRI_FILE_PATTERNS = [
    '*_t88_masked_gfc.hdr',
    '*_t88_gfc.hdr',
    '*.hdr'
]

# ============================================================================
# LOGGING
# ============================================================================

LOG_INTERVAL = 10  # Print every N batches
SAVE_CHECKPOINT_INTERVAL = 5  # Save every N epochs

# ============================================================================
# SUMMARY
# ============================================================================

def print_config():
    """Print configuration summary"""
    print("\n" + "="*80)
    print("KNOADNET CONFIGURATION")
    print("="*80)
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Dataset:          {DATASET}")
    print(f"Classes:          {CLASS_NAMES}")
    print(f"Device:           {DEVICE}")
    print(f"Image Size:       {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size:       {BATCH_SIZE}")
    print(f"Model:            {MODEL_CONFIG['vit_model']}")
    print(f"Epochs:           {NUM_EPOCHS}")
    print(f"CPU Threads:      {torch.get_num_threads()}")
    print("="*80 + "\n")