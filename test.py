from pathlib import Path

mri_root = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")

# Count .nii.gz files
nii_count = len(list(mri_root.glob("**/*.nii.gz")))
print(f"Total .nii.gz files: {nii_count}")

# Count subjects with .nii.gz
subjects_with_nii = set()
for f in mri_root.glob("**/*.nii.gz"):
    for parent in f.parents:
        if parent.name.startswith("OAS1_"):
            subjects_with_nii.add(parent.name)
            break

print(f"Subjects with .nii.gz: {len(subjects_with_nii)}")
print(f"\nBut phase1 only found: 100 subjects")
print(f"Missing: {len(subjects_with_nii) - 100} subjects!")