# Processing Alzheimer's Dataset (disc1-disc6)

## Quick Start

You now have **3 helper scripts** to process your data properly:

### Option 1: Run Everything Automatically (Recommended)
```bash
python RUN_ALL_STEPS.py
```
This interactive script will guide you through all 3 steps.

### Option 2: Run Steps Manually

#### Step 1: Convert HDR to NII (disc1-disc6 only)
```bash
python convert_hdr_disc1_to_disc6.py
```
- Converts `.hdr/.img` files to `.nii.gz` for disc1-disc6
- Skips already converted files
- Shows progress for each disc

#### Step 2: Process the Data
```bash
# First, fix the pipeline to use disc1-disc6
python fix_pipeline_for_disc1to6.py

# Then run the pipeline
python phase1_data_pipeline.py
```
- Fixes `phase1_data_pipeline.py` to use only disc1-disc6
- Preprocesses MRI images
- Creates train/val/test splits

#### Step 3: Train Baseline Model
```bash
python phase2_baselines.py
```
- Trains CNN/ResNet baseline models
- Evaluates on test set
- Saves results

## What Was Fixed?

### Problem
Your original code searched for MRI data but had issues:
1. It searched ALL disc folders (disc1-disc8) instead of just disc1-disc6
2. It had duplicate subjects (same subjects in root AND disc folders)
3. The deduplication logic didn't work properly

### Solution
The new code:
1. **Only searches disc1-disc6** (ignoring disc7-disc8 as you requested)
2. **Properly deduplicates** by subject name, not path
3. **Shows clear summary** of how many subjects per disc

## Expected Results

After running the scripts, you should have:

- **~112 unique subjects** from disc1-disc6
- **Train/val/test splits** saved in `data/splits/`
- **Processed MRI** slices in `data/processed/mri_slices/`
- **Model results** in `results/`
- **Visualizations** in `visualizations/`

## Folder Structure
```
data/
├── raw/
│   ├── mri/
│   │   ├── disc1/  (25 subjects)
│   │   ├── disc2/  (22 subjects)
│   │   ├── disc3/  (19 subjects)
│   │   ├── disc4/  (18 subjects)
│   │   ├── disc5/  (20 subjects)
│   │   ├── disc6/  (8 subjects)  
│   │   ├── disc7/  (ignored)
│   │   └── disc8/  (ignored)
│   └── tabular/
│       └── oasis_cross-sectional.csv
├── processed/
│   └── mri_slices/  (created by pipeline)
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Troubleshooting

### "nibabel not installed"
```bash
pip install nibabel
```

### "CSV file not found"
Make sure `data/raw/tabular/oasis_cross-sectional.csv` exists

### "No subjects found"
- Run Step 1 first to convert HDR files
- Check that disc1-disc6 folders exist in `data/raw/mri/`

### Pipeline fails
- Check that conversion (Step 1) completed successfully
- Verify .nii.gz files exist in disc folders
- Check logs for specific error messages

## Next Steps After Completion

1. Check `results/` for model performance
2. Review `visualizations/` for data distribution plots  
3. Analyze `logs/` for detailed processing information
4. Experiment with different model architectures

---

**Created**: 2025-12-02  
**Dataset**: OASIS-1 (disc1-disc6, ~112 subjects)  
**Goal**: Train baseline Alzheimer's classification model
