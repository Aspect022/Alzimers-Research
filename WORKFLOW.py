"""
STEP-BY-STEP WORKFLOW FOR ALZHEIMER'S DATASET PROCESSING
=========================================================

This guide will help you process disc1-disc6 data and run the baseline model.

STEP 1: Convert HDR to NII (disc1-disc6)
-----------------------------------------
Run the conversion script to convert .hdr/.img files to .nii.gz format
for discs 1 through 6 only.

Command:
    python convert_hdr_disc1_to_disc6.py

Expected output:
    - Will convert HDR/IMG files in disc1-disc6
    - Skip already converted files
    - Show summary of conversions per disc


STEP 2: Verify the Data
------------------------
Check how many subjects you have after conversion:

Command (PowerShell):
    foreach($disc in 1..6) { 
        Write-Host "disc$disc`: " -NoNewline
        (Get-ChildItem "data\raw\mri\disc$disc" -Directory -Filter "OAS1_*").Count 
    }

Expected:
    - disc1: ~25 subjects
    - disc2: ~22 subjects
    - disc3: ~19 subjects
    - disc4: ~18 subjects
    - disc5: ~20 subjects
    - disc6: ~8 subjects
    - Total: ~112 unique subjects


STEP 3: Run the Data Pipeline
------------------------------
The pipeline script has been fixed to use only disc1-disc6.

Command:
    python phase1_data_pipeline.py

What it does:
    - Loads clinical data from CSV
    - Discovers subjects from disc1-disc6
    - Preprocesses MRI images to 128x128
    - Creates train/val/test splits
    - Saves processed data


STEP 4: Run Baseline Models
----------------------------
After the pipeline completes successfully:

Command:
    python phase2_baselines.py

What it does:
    - Trains baseline models (CNN, ResNet, etc.)
    - Evaluates on test set
    - Saves results and visualizations


TROUBLESHOOTING
---------------
If step 1 fails:
    - Check nibabel is installed: pip install nibabel
    - Verify disc folders exist in data/raw/mri/

If step 3 fails:
    - Check CSV file exists: data/raw/tabular/oasis_cross-sectional.csv
    - Verify NII files were created in step 1
    - Check for error messages about missing files

If step 4 fails:
    - Ensure step 3 completed successfully
    - Check data/splits/ folder has train.csv, val.csv, test.csv
    - Verify PyTorch is installed


NEXT STEPS AFTER COMPLETION
----------------------------
1. Check results/ folder for model performance metrics
2. Check visualizations/ folder for plots
3. Check logs/ folder for detailed logs
4. Review data/splits/ to see train/val/test distribution
"""
print(__doc__)
