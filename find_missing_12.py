"""
Find why 112 MRI subjects become only 100 processed
"""
import pandas as pd
from pathlib import Path

csv_path = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\tabular\oasis_cross-sectional.csv")
mri_root = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")

df = pd.read_csv(csv_path)

# Find MRI subjects
subjects_with_mri = set()
disc_breakdown = {}

for disc_name in ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']:
    disc_path = mri_root / disc_name
    if disc_path.exists():
        subjects = [p.name for p in disc_path.glob('OAS1_*') if p.is_dir()]
        disc_breakdown[disc_name] = subjects
        subjects_with_mri.update(subjects)

print(f"Total unique MRI subjects: {len(subjects_with_mri)}")

# Check which have NIL files
subjects_with_nii = set()
for disc_name, subjects in disc_breakdown.items():
    for subj in subjects:
        disc_path = mri_root / disc_name / subj
        t88 = disc_path / 'PROCESSED' / 'MPRAGE' / 'T88_111'
        if t88.exists():
            nii_files = list(t88.glob('*.nii.gz'))
            if len(nii_files) > 0:
                subjects_with_nii.add(subj)

print(f"Subjects with .nii.gz: {len(subjects_with_nii)}")

# Missing nii
missing_nii = subjects_with_mri - subjects_with_nii
print(f"\nSubjects WITHOUT .nii.gz: {len(missing_nii)}")
if missing_nii:
    print(f"Examples: {list(missing_nii)[:10]}")

# Check if in clinical CSV
df_mri = df[df['ID'].isin(subjects_with_nii)]
print(f"\nSubjects with .nii.gz AND in clinical CSV: {len(df_mri)}")

# Missing from CSV
in_mri_not_csv = subjects_with_nii - set(df['ID'])
print(f"Subjects with .nii.gz but NOT in CSV: {len(in_mri_not_csv)}")
if in_mri_not_csv:
    print(f"Examples: {list(in_mri_not_csv)[:10]}")

# Diagnosis check
if len(df_mri) > 0:
    def map_cdr(cdr):
        if pd.isna(cdr):
            return None
        if cdr == 0.0:
            return 'CN'
        elif cdr == 0.5:
            return 'MCI'
        else:
            return 'AD'
    
    df_mri['diagnosis'] = df_mri['CDR'].apply(map_cdr)
    df_mri_valid = df_mri.dropna(subset=['diagnosis'])
    
    print(f"\nAfter dropping NA diagnosis: {len(df_mri_valid)}")
    print("\nDiagnosis distribution:")
    print(df_mri_valid['diagnosis'].value_counts())
