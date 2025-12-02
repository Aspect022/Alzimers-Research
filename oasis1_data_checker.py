"""
OASIS-1 Data Verification Script
=================================
Run this BEFORE notebook1 to verify your OASIS-1 data is correctly structured

Usage:
1. Update DATA_PATH to your local path
2. Run this script
3. Check output - should show found files
"""

import os
from pathlib import Path
import pandas as pd

# ============================================================================
# CONFIGURATION - UPDATE THIS PATH!
# ============================================================================
DATA_PATH = r"D:\Projects\AI-Projects\Alzimers\data\raw\mri\disc1"
CSV_PATH = r"D:\Projects\AI-Projects\Alzimers\data\raw\tabular\oasis_cross-sectional.csv"

# ============================================================================
# CHECK MRI FILES
# ============================================================================
print("=" * 80)
print("OASIS-1 DATA VERIFICATION")
print("=" * 80)

mri_dir = Path(DATA_PATH)

if not mri_dir.exists():
    print(f"❌ ERROR: MRI directory not found!")
    print(f"   Path: {mri_dir}")
    print(f"\n   Please update DATA_PATH in this script!")
    exit(1)

print(f"✓ MRI directory found: {mri_dir}\n")

# Find all subject directories
subject_dirs = list(mri_dir.glob('OAS1_*_MR1'))
print(f"Found {len(subject_dirs)} subject directories")

if len(subject_dirs) == 0:
    print("❌ ERROR: No subject directories found!")
    print("   Expected format: OAS1_XXXX_MR1")
    exit(1)

# Check first few subjects
print("\nChecking first 5 subjects:")
print("-" * 80)

found_files = []
for subject_dir in subject_dirs[:5]:
    subject_id = subject_dir.name
    print(f"\n{subject_id}:")
    
    # Check for PROCESSED/MPRAGE/T88_111 directory
    t88_dir = subject_dir / 'PROCESSED' / 'MPRAGE' / 'T88_111'
    
    if not t88_dir.exists():
        print(f"  ❌ T88_111 directory not found")
        continue
    else:
        print(f"  ✓ T88_111 directory exists")
    
    # Look for .hdr files (Analyze format)
    hdr_files = list(t88_dir.glob('*_t88_masked_gfc.hdr'))
    
    if len(hdr_files) == 0:
        hdr_files = list(t88_dir.glob('*_t88_gfc.hdr'))
    
    if len(hdr_files) == 0:
        hdr_files = list(t88_dir.glob('*.hdr'))
    
    if len(hdr_files) > 0:
        print(f"  ✓ Found .hdr file: {hdr_files[0].name}")
        
        # Check for corresponding .img file
        img_file = hdr_files[0].with_suffix('.img')
        if img_file.exists():
            print(f"  ✓ Found .img file: {img_file.name}")
            found_files.append({
                'subject_id': subject_id,
                'hdr_file': str(hdr_files[0]),
                'img_file': str(img_file)
            })
        else:
            print(f"  ❌ Missing .img file for {hdr_files[0].name}")
    else:
        print(f"  ❌ No .hdr files found")

print("\n" + "=" * 80)
print(f"SUMMARY: Found valid MRI files for {len(found_files)}/5 subjects checked")
print("=" * 80)

# ============================================================================
# CHECK CSV FILE
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING CLINICAL CSV FILE")
print("=" * 80)

csv_path = Path(CSV_PATH)

if not csv_path.exists():
    print(f"❌ ERROR: CSV file not found!")
    print(f"   Path: {csv_path}")
    print(f"\n   Download 'oasis_cross-sectional.csv' from OASIS website")
else:
    print(f"✓ CSV file found: {csv_path}")
    
    # Load and check CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded successfully")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_cols = ['ID', 'M/F', 'Age', 'MMSE', 'CDR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if len(missing_cols) == 0:
            print(f"  ✓ All required columns present")
        else:
            print(f"  ❌ Missing columns: {missing_cols}")
        
        # Show CDR distribution
        if 'CDR' in df.columns:
            print(f"\n  CDR Distribution:")
            print(df['CDR'].value_counts().sort_index())
            
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if len(found_files) > 0:
    print("✓ Your data looks good!")
    print("\nNext steps:")
    print("1. Upload data to Google Drive:")
    print("   - MRI files → /MyDrive/KnoADNet/data/raw/mri/")
    print("   - CSV file → /MyDrive/KnoADNet/data/raw/tabular/")
    print("\n2. Run notebook1_data_pipeline_OASIS1.py in Google Colab")
    print("   (The updated version handles .hdr/.img format automatically)")
else:
    print("❌ Issues found with data structure")
    print("\nPlease check:")
    print("1. OASIS-1 data is fully extracted")
    print("2. Folder structure matches: OAS1_XXXX_MR1/PROCESSED/MPRAGE/T88_111/")
    print("3. Both .hdr and .img files are present")

print("\n" + "=" * 80)