"""
Debug: Check clinical CSV vs MRI subjects
"""
import pandas as pd
from pathlib import Path
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load clinical CSV
csv_path = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\tabular\oasis_cross-sectional.csv")
mri_root = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")

if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Clinical CSV Stats:")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique subjects: {df['ID'].nunique()}")
    
    # Find MRI subjects
    subjects_with_mri = set()
    for disc in ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']:
        disc_path = mri_root / disc
        if disc_path.exists():
            for subject_dir in disc_path.glob('OAS1_*'):
                subjects_with_mri.add(subject_dir.name)
    
    print(f"\nMRI Subjects:")
    print(f"   With MRI files: {len(subjects_with_mri)}")
    
    # Match
    df_with_mri = df[df['ID'].isin(subjects_with_mri)]
    print(f"\nMatch:")
    print(f"   Clinical rows with MRI: {len(df_with_mri)}")
    
    # Class distribution
    if 'CDR' in df.columns:
        print(f"\nCDR Distribution (all clinical):")
        print(df['CDR'].value_counts().sort_index())
        
        print(f"\nCDR Distribution (with MRI):")
        print(df_with_mri['CDR'].value_counts().sort_index())
    
    # Missing MRI
    missing_mri = set(df['ID']) - subjects_with_mri
    print(f"\nSubjects in CSV but NO MRI: {len(missing_mri)}")
    if len(missing_mri) > 0 and len(missing_mri) <= 10:
        print(f"   Examples: {list(missing_mri)[:10]}")
        
else:
    print(f"CSV not found: {csv_path}")
