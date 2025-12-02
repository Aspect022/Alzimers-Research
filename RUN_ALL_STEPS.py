"""
======================================================================================
MASTER SCRIPT: Process Alzheimer's Dataset (disc1-disc6) and Run Baseline Model
======================================================================================

This script will guide you through 3 main steps:
1. Convert HDR files to NII format (disc1-disc6)
2. Fix and run the data pipeline  
3. Run the baseline model

Run this script and follow the prompts!
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Projects\A  I-Projects\Alzimers")

print("=" * 80)
print("ALZHEIMER'S DATASET PROCESSING - DISC1 TO DISC6")
print("=" * 80)

# STEP 1: Convert HDR to NII
print("\n" + "=" * 80)
print("STEP 1: Convert HDR/IMG files to NII.GZ (disc1-disc6)")
print("=" * 80)
print("\nThis will convert all .hdr/.img files to .nii.gz format")
print("for disc1 through disc6 only (skipping disc7-disc8)")

response = input("\nRun HDR to NII conversion? (y/n): ").lower()
if response == 'y':
    print("\nRunning conversion...")
    result = subprocess.run([sys.executable, "convert_hdr_disc1_to_disc6.py"], cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print("\n✓ Conversion complete!")
    else:
        print("\n✗ Conversion failed! Check errors above.")
        sys.exit(1)
else:
    print("Skipped conversion step.")

# STEP 2: Fix pipeline and run
print("\n" + "=" * 80)
print("STEP 2: Fix Pipeline and Process Data")
print("=" * 80)
print("\nThis will:")
print("  1. Update phase1_data_pipeline.py to use disc1-disc6 only")
print("  2. Load clinical data")
print("  3. Preprocess MRI images")
print("  4. Create train/val/test splits")

response = input("\nRun data pipeline? (y/n): ").lower()
if response == 'y':
    print("\nFixing pipeline...")
    result = subprocess.run([sys.executable, "fix_pipeline_for_disc1to6.py"], cwd=PROJECT_ROOT)
    
    print("\nRunning pipeline...")
    result = subprocess.run([sys.executable, "phase1_data_pipeline.py"], cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print("\n✓ Pipeline complete!")
    else:
        print("\n✗ Pipeline failed! Check errors above.")
        sys.exit(1)
else:
    print("Skipped pipeline step.")

# STEP 3: Run baseline model
print("\n" + "=" * 80)
print("STEP 3: Run Baseline Model Training")
print("=" * 80)
print("\nThis will train baseline models on the processed data")

response = input("\nRun baseline model training? (y/n): ").lower()
if response == 'y':
    print("\nRunning baseline training...")
    result = subprocess.run([sys.executable, "phase2_baselines.py"], cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print("\n✓ Baseline training complete!")
    else:
        print("\n✗ Training failed! Check errors above.")
        sys.exit(1)
else:
    print("Skipped baseline training.")

print("\n" + "=" * 80)
print("ALL STEPS COMPLETE!")
print("=" * 80)
print("\nCheck these folders for results:")
print(f"  - Processed data: {PROJECT_ROOT}/data/processed")
print(f"  - Train/val/test splits: {PROJECT_ROOT}/data/splits")
print(f"  - Model results: {PROJECT_ROOT}/results")
print(f"  - Visualizations: {PROJECT_ROOT}/visualizations")
print(f"  - Logs: {PROJECT_ROOT}/logs")
