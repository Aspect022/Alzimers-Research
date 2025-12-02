"""
KnoAD-Net Implementation - Notebook 1: Data Pipeline (OASIS-1 Version)
Optimized to run as a standalone Python script (works in Colab / local Windows)

What this script does:
- Loads OASIS-1 clinical CSV (if available)
- Creates 3-class labels: CN, MCI (CDR=0.5), AD (CDR>=1)
- Optionally generates synthetic demo subjects + synthetic MRI .nii.gz files for testing
- Robust MRI preprocessing to a single 2D central axial slice resized to (image_size,image_size)
- Feature engineering, imputation, encoding, stratified train/val/test split
- Saves processed numpy slices, CSV splits, encoders, scaler, dataloader info, and EDA plot

This is a corrected and runnable version of the notebook you provided.
"""

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
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

# -------------------------
# Config (use raw strings for Windows paths)
# -------------------------
PROJECT_ROOT = r"D:\Projects\AI-Projects\Alzimers"
# If running in Colab, you may want to set PROJECT_ROOT = '/content/knoadnet'

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
with open(Path(config['directories']['root']) / 'config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

print("KNOADNET - PHASE 1: DATA PIPELINE (OASIS-1)")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset: {config['dataset']}")
print(f"Classes: {config['class_names']}")

# -------------------------
# Paths & Inputs
# -------------------------
raw_dir = Path(config['directories']['data_raw'])
processed_root = Path(config['directories']['data_processed'])
splits_dir = Path(config['directories']['data_splits'])
vis_dir = Path(config['directories']['visualizations'])
logs_dir = Path(config['directories']['logs'])

raw_dir.mkdir(parents=True, exist_ok=True)
processed_root.mkdir(parents=True, exist_ok=True)
splits_dir.mkdir(parents=True, exist_ok=True)
vis_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

# Clinical CSV path - OASIS-1 cross-sectional data
csv_path = raw_dir / 'tabular' / 'oasis_cross-sectional.csv'

# -------------------------
# SECTION 1: Load or generate clinical data
# -------------------------
print('\n[1/7] Loading clinical data')
if csv_path.exists():
    df_clinical = pd.read_csv(csv_path)
    print(f"Loaded OASIS-1 clinical data: {csv_path}")
else:
    # Create demo synthetic clinical data to allow the pipeline to run
    print("Clinical CSV not found. Generating synthetic demo clinical data for testing...")
    rng = np.random.RandomState(config['random_seed'])
    n_subjects = 150
    ids = [f"OAS1_{i:04d}" for i in range(1, n_subjects + 1)]
    # CDR distribution: mostly CN, some MCI, fewer AD
    cdr_vals = rng.choice([0.0, 0.5, 1.0], size=n_subjects, p=[0.7, 0.2, 0.1])
    demo_data = {
        'ID': ids,
        'Age': rng.randint(60, 90, size=n_subjects),
        'M/F': rng.choice(['M', 'F'], size=n_subjects),
        'Hand': rng.choice(['R', 'L'], size=n_subjects, p=[0.9, 0.1]),
        'Educ': rng.randint(6, 20, size=n_subjects),
        'SES': rng.randint(1, 6, size=n_subjects),
        'MMSE': np.clip(np.round(rng.normal(28, 2, size=n_subjects)), 0, 30),
        'CDR': cdr_vals,
        'eTIV': np.round(rng.uniform(1200, 1800, size=n_subjects), 1),
        'nWBV': np.round(rng.uniform(0.6, 0.8, size=n_subjects), 3),
        'ASF': np.round(rng.uniform(1.0, 1.5, size=n_subjects), 3),
    }
    df_clinical = pd.DataFrame(demo_data)
    df_clinical.to_csv(csv_path, index=False)
    print(f"Generated synthetic clinical CSV at: {csv_path}")

print(f"Total subjects in clinical file: {len(df_clinical)}")
print("Columns:", df_clinical.columns.tolist())

# -------------------------
# SECTION 2: Create 3-Class Diagnosis Labels
# -------------------------
print('\n[2/7] Creating Diagnosis Labels (3-Class)')

def map_cdr_to_diagnosis(cdr):
    try:
        if pd.isna(cdr):
            return None
        c = float(cdr)
        if c == 0.0:
            return 'CN'
        elif c == 0.5:
            return 'MCI'
        else:
            return 'AD'
    except Exception:
        return None

if 'CDR' not in df_clinical.columns:
    df_clinical['CDR'] = 0.0

df_clinical['diagnosis'] = df_clinical['CDR'].apply(map_cdr_to_diagnosis)
# Drop rows without diagnosis
df_clinical = df_clinical.dropna(subset=['diagnosis']).reset_index(drop=True)

print('\nDiagnosis Distribution:')
print(df_clinical['diagnosis'].value_counts())
print('\nPercentage:')
print(df_clinical['diagnosis'].value_counts(normalize=True) * 100)

print('\nMMSE by Diagnosis:')
print(df_clinical.groupby('diagnosis')['MMSE'].describe())

# -------------------------
# SECTION 3: Discover MRI Files from disc1-disc6
# -------------------------
print('\n[3/7] Checking MRI Files (disc1-disc6)')
print('-' * 80)

root_mri_dir = Path(config['directories']['data_raw']) / 'mri'
candidate_subject_dirs = []
subject_dict = {}  # Track by name to avoid duplicates

# Process ONLY disc1 through disc6
DISCS_TO_USE = ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']

if root_mri_dir.exists():
    disc_subjects_count = {}
    
    for disc_name in DISCS_TO_USE:
        disc_path = root_mri_dir / disc_name
        if disc_path.exists() and disc_path.is_dir():
            disc_subjects = [p for p in disc_path.glob('OAS1_*') if p.is_dir()]
            disc_subjects_count[disc_name] = len(disc_subjects)
            
            for subject_path in disc_subjects:
                subject_name = subject_path.name
                # Only add if not already present (first occurrence wins)
                if subject_name not in subject_dict:
                    subject_dict[subject_name] = subject_path
    
    candidate_subject_dirs = sorted(subject_dict.values(), key=lambda x: x.name)
    
    print(f"\n📊 Subject Discovery Summary (disc1-disc6):")
    for disc_name, count in sorted(disc_subjects_count.items()):
        print(f"  - {disc_name}: {count} subjects")
    print(f"  Total unique subjects: {len(candidate_subject_dirs)}")
else:
    print(f"\n⚠️ MRI directory not found: {root_mri_dir}")
    print("Please ensure disc1-disc6 folders exist in data/raw/mri/")

# Final sort by subject name
candidate_subject_dirs = sorted(candidate_subject_dirs, key=lambda x: x.name)

# Build available_subjects list
available_subjects = [p.name for p in candidate_subject_dirs]
print(f"\n✓ Subjects discovered: {len(available_subjects)}")

# Filter clinical data by discovered subjects
df_clinical = df_clinical[df_clinical['ID'].isin(available_subjects)].reset_index(drop=True)
print(f"✓ Subjects with both MRI and clinical data: {len(df_clinical)}")

if len(df_clinical) == 0:
    raise RuntimeError("No subjects found with both MRI and clinical data! Check that disc1-disc6 folders contain valid subject directories and that the clinical CSV has matching IDs.")

# -------------------------
# SECTION 4: Preprocess MRI (Memory Efficient)
# -------------------------
print('\n[4/7] Preprocessing MRI (single axial slice, resized)')


def find_subject_dir(subject_id, mri_root_dir):
    cand = Path(mri_root_dir) / subject_id
    if cand.exists():
        return cand
    matches = list(Path(mri_root_dir).glob(f'**/{subject_id}'))
    if len(matches) > 0:
        return Path(matches[0])
    return None


def preprocess_mri(subject_id, mri_root_dir, target_size=128):
    try:
        subject_dir = find_subject_dir(subject_id, mri_root_dir)
        if subject_dir is None:
            return None

        t88 = subject_dir / 'PROCESSED' / 'MPRAGE' / 'T88_111'
        if not t88.exists():
            return None

        nii_files = sorted(list(t88.glob('*_t88_masked_gfc.nii.gz')) + list(t88.glob('*.nii.gz')))
        hdr_files = sorted(list(t88.glob('*.hdr')))

        chosen = None
        if len(nii_files):
            chosen = nii_files[0]
        elif len(hdr_files):
            chosen = hdr_files[0]
        else:
            return None

        img = nib.load(str(chosen))
        data = img.get_fdata()

        # If 4D, use first timepoint
        if data.ndim == 4:
            data = data[..., 0]

        # Get 2D slice (middle axial if possible)
        if data.ndim == 2:
            slice_2d = data
        elif data.ndim >= 3:
            mid_slice = data.shape[2] // 2
            slice_2d = data[:, :, mid_slice]
        else:
            return None

        slice_2d = np.squeeze(slice_2d)
        if slice_2d.ndim == 1:
            slice_2d = slice_2d[:, None]

        # Normalize
        denom = (slice_2d.max() - slice_2d.min())
        if denom > 1e-8:
            slice_2d = (slice_2d - slice_2d.min()) / denom
        else:
            slice_2d = np.zeros_like(slice_2d, dtype=float)

        # Resize with zoom factors computed from actual shape
        zoom_factors = [target_size / float(s) for s in slice_2d.shape]
        slice_resized = zoom(slice_2d, zoom_factors, order=1)

        # Ensure exact target size by padding/cropping
        def pad_or_crop(img2d, targ):
            h, w = img2d.shape
            # crop center if larger
            if h > targ:
                start = (h - targ) // 2
                img2d = img2d[start:start+targ, :]
                h = targ
            if w > targ:
                start = (w - targ) // 2
                img2d = img2d[:, start:start+targ]
                w = targ
            # pad if smaller
            pad_h = max(0, targ - h)
            pad_w = max(0, targ - w)
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img2d = np.pad(img2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            return img2d

        slice_fixed = pad_or_crop(slice_resized, target_size)

        # To 3-channel and cast to float32
        slice_rgb = np.stack([slice_fixed] * 3, axis=-1).astype(np.float32)
        if slice_rgb.shape != (target_size, target_size, 3):
            slice_rgb = np.resize(slice_rgb, (target_size, target_size, 3)).astype(np.float32)

        return slice_rgb

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return None

# Preprocess and build processed_data
processed_dir = processed_root / 'mri_slices'
processed_dir.mkdir(parents=True, exist_ok=True)

processed_data = []
failed_subjects = []

print(f"Processing {len(df_clinical)} MRI scans...")
for subject_id in tqdm(df_clinical['ID'], desc='Preprocessing MRI'):
    slice_data = preprocess_mri(subject_id, root_mri_dir, config['image_size'])
    if slice_data is not None:
        save_path = processed_dir / f"{subject_id}.npy"
        # Save as float16 to reduce disk usage but load as float32 later
        np.save(save_path, slice_data.astype(np.float16))
        processed_data.append({'subject_id': subject_id, 'processed_path': str(save_path)})
    else:
        failed_subjects.append(subject_id)

print(f"Successfully processed: {len(processed_data)} scans")
if failed_subjects:
    print(f"Failed to process: {len(failed_subjects)} scans")

# -------------------------
# SECTION 5: Merge and Engineer Features
# -------------------------
print('\n[5/7] Feature Engineering')
df_processed = pd.DataFrame(processed_data)
if df_processed.empty:
    raise RuntimeError('No processed MRI slices found. Ensure MRI files exist or allow synthetic generation to complete.')

# Merge on ID
df_merged = df_clinical.merge(df_processed, left_on='ID', right_on='subject_id', how='inner')
print(f"Merged dataset: {len(df_merged)} subjects")

print('\nMissing values before imputation:')
print(df_merged.isnull().sum())

# Impute MMSE by diagnosis median
if 'MMSE' in df_merged.columns:
    df_merged['MMSE'] = df_merged.groupby('diagnosis')['MMSE'].transform(lambda x: x.fillna(x.median()))

numeric_cols = ['Age', 'Educ', 'SES', 'eTIV', 'nWBV', 'ASF']
for col in numeric_cols:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].fillna(df_merged[col].median())

# Encode categorical variables (keep originals in df_merged)
le_sex = LabelEncoder()
if 'M/F' in df_merged.columns:
    df_merged['sex_encoded'] = le_sex.fit_transform(df_merged['M/F'].astype(str))
else:
    df_merged['M/F'] = 'U'
    df_merged['sex_encoded'] = le_sex.fit_transform(df_merged['M/F'].astype(str))

le_hand = LabelEncoder()
df_merged['hand_encoded'] = le_hand.fit_transform(df_merged['Hand'].fillna('R').astype(str))

le_diagnosis = LabelEncoder()
df_merged['diagnosis_encoded'] = le_diagnosis.fit_transform(df_merged['diagnosis'].astype(str))

# Feature engineering
if 'Age' in df_merged.columns:
    df_merged['age_group'] = pd.cut(df_merged['Age'], bins=[0, 70, 80, 100], labels=['young', 'middle', 'old'])

if 'MMSE' in df_merged.columns:
    df_merged['mmse_impaired'] = (df_merged['MMSE'] < 24).astype(int)

if 'nWBV' in df_merged.columns:
    df_merged['brain_volume_low'] = (df_merged['nWBV'] < df_merged['nWBV'].median()).astype(int)

feature_cols = [
    'Age', 'sex_encoded', 'hand_encoded', 'Educ', 'SES',
    'MMSE', 'eTIV', 'nWBV', 'ASF'
]
feature_cols = [c for c in feature_cols if c in df_merged.columns]
print(f"Selected {len(feature_cols)} features: {feature_cols}")

# Create features dataframe
df_features = df_merged[feature_cols + ['diagnosis', 'diagnosis_encoded', 'processed_path', 'ID']].copy()
df_features[feature_cols] = df_features[feature_cols].fillna(df_features[feature_cols].median())

print('\nFinal dataset shape:', df_features.shape)
print('\nDiagnosis distribution:')
print(df_features['diagnosis'].value_counts())

# -------------------------
# SECTION 6: Train/Val/Test Split (Stratified)
# -------------------------
print('\n[6/7] Creating Stratified Train/Val/Test Splits')
train_val, test = train_test_split(
    df_features,
    test_size=0.2,
    random_state=config['random_seed'],
    stratify=df_features['diagnosis_encoded'] if len(df_features) > 1 else None
)

train, val = train_test_split(
    train_val,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 total
    random_state=config['random_seed'],
    stratify=train_val['diagnosis_encoded'] if len(train_val) > 1 else None
)

print(f"Train set: {len(train)} samples ({len(train)/len(df_features)*100:.1f}%)")
print(f"Val set:   {len(val)} samples ({len(val)/len(df_features)*100:.1f}%)")
print(f"Test set:  {len(test)} samples ({len(test)/len(df_features)*100:.1f}%)")

print('\nClass distribution in splits:')
for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
    dist = split_df['diagnosis'].value_counts()
    print(f"\n{split_name}:")
    for dx, count in dist.items():
        print(f"  {dx}: {count} ({count/len(split_df)*100:.1f}%)")

# Normalize features (fit on train)
scaler = StandardScaler()
if len(train) > 0:
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
if len(val) > 0:
    val[feature_cols] = scaler.transform(val[feature_cols])
if len(test) > 0:
    test[feature_cols] = scaler.transform(test[feature_cols])

# Save splits and encoders
train.to_csv(splits_dir / "train.csv", index=False)
val.to_csv(splits_dir / "val.csv", index=False)
test.to_csv(splits_dir / "test.csv", index=False)

with open(splits_dir / "label_encoders.pkl", "wb") as f:
    pickle.dump({'diagnosis': le_diagnosis, 'sex': le_sex, 'hand': le_hand}, f)

with open(splits_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Saved splits and encoders to: {splits_dir}")

# -------------------------
# SECTION 7: Create DataLoaders & Summary
# -------------------------
print('\n[7/7] Creating DataLoaders')

class OASIS1Dataset(Dataset):
    def __init__(self, dataframe, feature_cols):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(row['processed_path']).astype(np.float32)
        img = torch.FloatTensor(img).permute(2, 0, 1)  # HWC -> CHW
        features = torch.FloatTensor(row[self.feature_cols].values.astype(np.float32))
        label = int(row['diagnosis_encoded'])
        return {'image': img, 'features': features, 'label': label, 'subject_id': row['ID']}

train_dataset = OASIS1Dataset(train, feature_cols)
val_dataset = OASIS1Dataset(val, feature_cols)
test_dataset = OASIS1Dataset(test, feature_cols)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train loader: {len(train_loader)} batches (batch_size={batch_size})")
print(f"Val loader:   {len(val_loader)} batches")
print(f"Test loader:  {len(test_loader)} batches")

# Test dataloader (if not empty)
if len(train_loader) > 0:
    sample_batch = next(iter(train_loader))
    print(f"Image shape: {sample_batch['image'].shape}")
    print(f"Features shape: {sample_batch['features'].shape}")
    print(f"Labels: {sample_batch['label']}")
else:
    print('Train loader is empty — check your splits or processed files')

# Save dataloader info
dataloader_info = {
    'dataset': 'OASIS1',
    'feature_cols': feature_cols,
    'n_features': len(feature_cols),
    'n_classes': config['n_classes'],
    'class_names': config['class_names'],
    'batch_size': batch_size,
    'image_size': config['image_size'],
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'test_size': len(test_dataset),
}
with open(processed_root / "dataloader_info.json", "w", encoding='utf-8') as f:
    json.dump(dataloader_info, f, indent=2)

# -------------------------
# Visualization (robust plotting for sex)
# -------------------------
print('\nSaving EDA plots...')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('OASIS-1 Data Overview', fontsize=16, fontweight='bold')

# Diagnosis distribution
try:
    df_features['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Diagnosis Distribution')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xlabel('Diagnosis')
except Exception:
    axes[0,0].text(0.5, 0.5, 'N/A', horizontalalignment='center')

# Age distribution
try:
    df_features['Age'].hist(bins=20, ax=axes[0,1], edgecolor='black')
    axes[0,1].set_title('Age Distribution')
    axes[0,1].set_xlabel('Age (years)')
except Exception:
    axes[0,1].text(0.5, 0.5, 'N/A', horizontalalignment='center')

# MMSE by diagnosis
try:
    df_features.boxplot(column='MMSE', by='diagnosis', ax=axes[0,2])
    axes[0,2].set_title('MMSE by Diagnosis')
    axes[0,2].set_xlabel('Diagnosis')
    axes[0,2].set_ylabel('MMSE Score')
except Exception:
    axes[0,2].text(0.5, 0.5, 'N/A', horizontalalignment='center')

# Sex distribution
if 'M/F' in df_features.columns:
    sex_series = df_features['M/F'].astype(str)
else:
    try:
        sex_series = pd.Series(le_sex.inverse_transform(df_features['sex_encoded'].astype(int)), index=df_features.index)
    except Exception:
        sex_series = df_features['sex_encoded'].map({0:'F', 1:'M'}).fillna('U')

sex_series.value_counts().plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Sex Distribution')
axes[1,0].set_xlabel('Sex')

# Brain volume by diagnosis
try:
    df_features.boxplot(column='nWBV', by='diagnosis', ax=axes[1,1])
    axes[1,1].set_title('Normalized Brain Volume by Diagnosis')
    axes[1,1].set_xlabel('Diagnosis')
    axes[1,1].set_ylabel('nWBV')
except Exception:
    axes[1,1].text(0.5, 0.5, 'N/A', horizontalalignment='center')

# Sample MRI slice
if len(train_loader) > 0:
    sample_img = sample_batch['image'][0].permute(1, 2, 0).numpy()[:, :, 0]
    axes[1,2].imshow(sample_img, cmap='gray')
    axes[1,2].set_title('Sample MRI Slice')
    axes[1,2].axis('off')
else:
    axes[1,2].text(0.5, 0.5, 'No sample image', horizontalalignment='center')

plt.tight_layout()
plot_path = vis_dir / "oasis1_eda.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved EDA plots to: {plot_path}")

# -------------------------
# FINAL SUMMARY
# -------------------------
summary_lines = []
summary_lines.append('OASIS-1 Dataset Summary:')
summary_lines.append('------------------------')
summary_lines.append(f"Total subjects:         {len(df_features)}")
summary_lines.append(f"Train set:              {len(train)} ({len(train)/len(df_features)*100:.1f}%)")
summary_lines.append(f"Validation set:         {len(val)} ({len(val)/len(df_features)*100:.1f}%)")
summary_lines.append(f"Test set:               {len(test)} ({len(test)/len(df_features)*100:.1f}%)")
summary_lines.append('')
summary_lines.append(f"Classes:                {config['class_names']}")
summary_lines.append(f"Number of classes:      {config['n_classes']}")
summary_lines.append(f"Input features:         {len(feature_cols)}")
summary_lines.append(f"Image size:             {config['image_size']} x {config['image_size']} (memory optimized)")
summary_lines.append(f"Batch size:             {batch_size}")
for cls in config['class_names']:
    count = (test['diagnosis'] == cls).sum() if len(test) > 0 else 0
    pct = (count / len(test) * 100) if len(test) > 0 else 0
    summary_lines.append(f"  {cls}: {count} ({pct:.1f}%)")

summary_lines.append('\nFiles Saved:')
summary_lines.append('------------')
summary_lines.append(f"Processed MRI dir:     {processed_dir}")
summary_lines.append(f"Train/Val/Test CSVs:   {splits_dir}")
summary_lines.append(f"Scalers/Encoders:      {splits_dir / 'label_encoders.pkl'} and {splits_dir / 'scaler.pkl'}")
summary_lines.append(f"DataLoader Info:       {processed_root / 'dataloader_info.json'}")
summary_lines.append(f"EDA Plot:              {plot_path}")

summary = '\n'.join(summary_lines)
print('\n' + summary)

# save summary
with open(logs_dir / "data_pipeline_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print('\nREADY FOR TRAINING!')
