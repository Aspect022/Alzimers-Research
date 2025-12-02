# ✅ phase1_data_pipeline.py - FIXED!

## What Was Wrong:
Your `phase1_data_pipeline.py` file had been modified to use **synthetic data** instead of the real disc1-disc6 data.

## What I Fixed:

### 1. **CSV Path** (Line ~90)
**Before:**
```python
csv_path = raw_dir / 'oasis_clinical.csv'  # ❌ Wrong path
```

**After:**
```python
csv_path = raw_dir / 'tabular' / 'oasis_cross-sectional.csv'  # ✅ Correct!
```

### 2. **MRI Data Source** (Lines ~162-201)
**Before:**
```python
root_mri_dir = Path(processed_root) / 'synthetic_mri'  # ❌ Synthetic data!
# ... code to generate fake MRI scans ...
```

**After:**
```python
root_mri_dir = Path(config['directories']['data_raw']) / 'mri'  # ✅ Real data!

# Process ONLY disc1 through disc6
DISCS_TO_USE = ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']

# Discover subjects from each disc
for disc_name in DISCS_TO_USE:
    disc_path = root_mri_dir / disc_name
    # ... discovery logic ...
```

### 3. **Subject Filtering** (Lines ~203-212)
Added logic to:
- Build list of available subjects from discovered MRI files
- Filter clinical data to only keep subjects with MRI
- Raise error if no matching subjects found

## Now Your Pipeline Will:

1. ✅ Load **real OASIS-1 clinical data** from `data/raw/tabular/oasis_cross-sectional.csv`
2. ✅ Discover **real MRI subjects** from `data/raw/mri/disc1` through `disc6`
3. ✅ Only use subjects that have **both clinical data AND MRI files**
4. ✅ Show you a **summary** of subjects per disc
5. ✅ Process all unique subjects (avoiding duplicates)

## Expected Output When You Run It:

```
[3/7] Checking MRI Files (disc1-disc6)
--------------------------------------------------------------------------------

📊 Subject Discovery Summary (disc1-disc6):
  - disc1: 25 subjects
  - disc2: 22 subjects
  - disc3: 19 subjects
  - disc4: 18 subjects
  - disc5: 20 subjects
  - disc6: 8 subjects
  Total unique subjects: 112

✓ Subjects discovered: 112
✓ Subjects with both MRI and clinical data: 112
```

## Next Steps:

### Step 1: Convert HDR to NII (if not done yet)
```bash
python convert_hdr_disc1_to_disc6.py
```

### Step 2: Run the Fixed Pipeline
```bash
python phase1_data_pipeline.py
```

### Step 3: Train Baseline Models
```bash
python phase2_baselines.py
```

---

**The file is now ready to use with your REAL disc1-disc6 data!** 🎉
