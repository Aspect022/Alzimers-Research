"""
KnoAD-Net Implementation - Notebook 1: Data Pipeline (OASIS-1 Version)
======================================================================
OPTIMIZED FOR FREE COLAB - OASIS-1 Dataset

This script handles:
- OASIS-1 data loading
- MRI preprocessing (smaller images for memory efficiency)
- 3-class classification: CN, MCI (CDR 0.5), AD (CDR 1+)
- Tabular data cleaning
- Train/val/test splitting
"""

# ========================================================================
# SECTION 1: Setup & Imports
# ========================================================================
print("=" * 80)
print("KNO ADNET - PHASE 1: DATA PIPELINE (OASIS-1)")
print("=" * 80)

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Config (use raw strings for Windows paths)
# -------------------------
PROJECT_ROOT = r"D:\Projects\AI-Projects\Alzimers"
sys.path.append(PROJECT_ROOT)

config = {
    'project_root': PROJECT_ROOT,
    'dataset': 'OASIS1',
    'n_classes': 3,  # CN, MCI, AD
    'class_names': ['CN', 'MCI', 'AD'],
    'random_seed': 42,
    'image_size': 128,  # Reduced from 224 for memory efficiency
    'directories': {
        'root': PROJECT_ROOT,
        'data_raw': f'{PROJECT_ROOT}/data/raw',
        'data_processed': f'{PROJECT_ROOT}/data/processed',
        'data_splits': f'{PROJECT_ROOT}/data/splits',
        'models': f'{PROJECT_ROOT}/models',
        'checkpoints': f'{PROJECT_ROOT}/checkpoints',
        'results': f'{PROJECT_ROOT}/results',
        'logs': f'{PROJECT_ROOT}/logs',
        'visualizations': f'{PROJECT_ROOT}/visualizations',
        'rag_knowledge': f'{PROJECT_ROOT}/rag_knowledge',
    }
}

# create directories if missing
for k, p in config['directories'].items():
    Path(p).mkdir(parents=True, exist_ok=True)

# Save a JSON copy of config (optional)
with open(Path(config['directories']['root']) / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Project root: {PROJECT_ROOT}")
print(f"✓ Dataset: {config['dataset']}")
print(f"✓ Classes: {config['class_names']}")
print(f"âœ" Image size: {config['image_size']}x{config['image_size']} (memory optimized)")

# set seed
np.random.seed(config['random_seed'])
torch.manual_seed(config['random_seed'])

# ========================================================================
# SECTION 2: Load OASIS-1 Clinical Data
# ========================================================================
print("\n[1/7] Loading OASIS-1 Clinical Data")
print("-" * 80)

csv_path = Path(config['directories']['data_raw']) / 'tabular' / 'oasis_cross-sectional.csv'

if not csv_path.exists():
    print("\n⚠️  OASIS-1 CSV not found!")
    print(f"Expected location: {csv_path}")
    print("\nFor now, generating DEMO data to test pipeline...")
    n_subjects = 200
    demo_data = {
        '

ID': [f'OAS1_{i:04d}_MR1' for i in range(1, n_subjects+1)],
        'M/F': np.random.choice(['M', 'F'], n_subjects),
        'Hand': np.random.choice(['R', 'L'], n_subjects, p=[0.9, 0.1]),
        'Age': np.random.randint(60, 95, n_subjects),
        'Educ': np.random.randint(8, 20, n_subjects),
        'SES': np.random.randint(1, 6, n_subjects),
        'MMSE': np.random.randint(15, 30, n_subjects),
        'CDR': np.random.choice([0, 0.5, 1, 2], n_subjects, p=[0.40, 0.30, 0.20, 0.10]),
        'eTIV': np.random.randint(1200, 1800, n_subjects),
        'nWBV': np.random.uniform(0.6, 0.8, n_subjects),
        'ASF': np.random.uniform(1.0, 1.5, n_subjects)
    }
    df_clinical = pd.DataFrame(demo_data)
    print(f"✓ Generated {n_subjects} demo subjects")
else:
    df_clinical = pd.read_csv(csv_path)
    print(f"✓ Loaded OASIS-1 clinical data: {csv_path}")
    print(f"✓ Total subjects: {len(df_clinical)}")

print("\nDataset columns:", df_clinical.columns.tolist())
print("\nFirst few rows:")
print(df_clinical.head())

# ========================================================================
# SECTION 3: Create 3-Class Diagnosis Labels
# ========================================================================
print("\n[2/7] Creating Diagnosis Labels (3-Class)")
print("-" * 80)

def map_cdr_to_diagnosis(cdr):
    if pd.isna(cdr):
        return None
    elif cdr == 0:
        return 'CN'
    elif cdr == 0.5:
        return 'MCI'
    else:
        return 'AD'

df_clinical['diagnosis'] = df_clinical['CDR'].apply(map_cdr_to_diagnosis)
df_clinical = df_clinical.dropna(subset=['diagnosis']).reset_index(drop=True)

print("\nDiagnosis Distribution:")
print(df_clinical['diagnosis'].value_counts())
print("\nPercentage:")
print(df_clinical['diagnosis'].value_counts(normalize=True) * 100)

print("\nMMSE by Diagnosis:")
print(df_clinical.groupby('diagnosis')['MMSE'].describe())

# ========================================================================
# SECTION 4: Checking MRI Files (search across discs) - FIXED VERSION
# ========================================================================
print("\n[3/7] Checking MRI Files")
print("-" * 80)

root_mri_dir = Path(config['directories']['data_raw']) / 'mri'
candidate_subject_dirs = []
subject_dict = {}  # Use dict to track subjects by name and prioritize disc folders

if root_mri_dir.exists():
    # First, collect subjects from disc* subdirs (disc1..disc6) - these take priority
    disc_subjects_count = {}
    for sub in sorted(root_mri_dir.iterdir()):
        if sub.is_dir() and sub.name.lower().startswith('disc'):
            disc_subjects = [p for p in sub.glob('OAS1_*') if p.is_dir()]
            disc_subjects_count[sub.name] = len(disc_subjects)
            for subject_path in disc_subjects:
                subject_name = subject_path.name
                # Only add if not already present (first disc wins if subject in multiple discs)
                if subject_name not in subject_dict:
                    subject_dict[subject_name] = subject_path
    
    # Then add root-level OAS1_* subjects (only if not already in disc folders)
    root_subjects = [p for p in root_mri_dir.glob('OAS1_*') if p.is_dir()]
    root_count = 0
    for subject_path in root_subjects:
        subject_name = subject_path.name
        if subject_name not in subject_dict:
            subject_dict[subject_name] = subject_path
            root_count += 1
    
    candidate_subject_dirs = sorted(subject_dict.values(), key=lambda x: x.name)
    
    print(f"\n📊 Subject Discovery Summary:")
    if disc_subjects_count:
        print(f"  Subjects from disc folders:")
        for disc_name, count in sorted(disc_subjects_count.items()):
            print(f"    - {disc_name}: {count} subjects")
        print(f"  Subjects from root (no disc duplicate): {root_count}")
    else:
        print(f"  Subjects from root: {len(root_subjects)}")

# Final sort
candidate_subject_dirs = sorted(candidate_subject_dirs, key=lambda x: x.name)

if len(candidate_subject_dirs) > 0:
    print(f"✓ Found MRI subjects across MRI root: {root_mri_dir}")
    print(f"  Total unique subject directories: {len(candidate_subject_dirs)}")
    mri_root_for_search = root_mri_dir
else:
    # fallback: generate synthetic dataset under root_mri_dir
    print("\n⚠️  No MRI files found under:", root_mri_dir)
    print("Expected structure: <mri_root>/disc1/OAS1_XXXX_MR1/... or <mri_root>/OAS1_XXXX_MR1/...")
    print("\nGenerating SYNTHETIC MRI data for pipeline testing...")
    root_mri_dir.mkdir(parents=True, exist_ok=True)
    for idx, subject_id in enumerate(tqdm(df_clinical['ID'].head(100), desc="Creating synthetic MRI")):
        subject_dir = root_mri_dir / subject_id / 'PROCESSED' / 'MPRAGE' / 'T88_111'
        subject_dir.mkdir(exist_ok=True, parents=True)
        volume = np.random.randn(64, 64, 64).astype(np.float32) * 100 + 500
        center = np.array([32, 32, 32])
        xx, yy, zz = np.mgrid[0:64, 0:64, 0:64]
        dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2)
        brain_mask = dist < 25
        volume[brain_mask] += 200
        img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(img, str(subject_dir / f'{subject_id}_mpr_n4_anon_111_t88_masked_gfc.nii.gz'))
    # re-scan
    candidate_subject_dirs = sorted([p for p in root_mri_dir.glob('OAS1_*') if p.is_dir()])
    print(f"\n✓ Generated synthetic MRI for {len(candidate_subject_dirs)} subjects")
    mri_root_for_search = root_mri_dir

# Build available_subjects
available_subjects = [p.name for p in candidate_subject_dirs]
print(f"\n✓ Subjects discovered: {len(available_subjects)}")

# Filter clinical data by discovered subjects
df_clinical = df_clinical[df_clinical['ID'].isin(available_subjects)].reset_index(drop=True)
print(f"\n✓ Subjects with both MRI and clinical data: {len(df_clinical)}")

# Continue with rest of file...
print("\n✓✓✓ DATA DISCOVERY COMPLETE - Continue with preprocessing...")
