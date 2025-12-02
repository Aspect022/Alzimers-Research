"""Test script to debug disc discovery"""
from pathlib import Path

root_mri_dir = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")
candidate_subject_dirs = []

if root_mri_dir.exists():
    # OAS1_* directly under root
    root_subjects = [p for p in root_mri_dir.glob('OAS1_*') if p.is_dir()]
    print(f"Subjects directly under root: {len(root_subjects)}")
    candidate_subject_dirs += root_subjects
    
    # OAS1_* under any disc* subdir (disc1..disc6)
    disc_subjects_by_disc = {}
    for sub in sorted(root_mri_dir.iterdir()):
        if sub.is_dir() and sub.name.lower().startswith('disc'):
            disc_subjects = [p for p in sub.glob('OAS1_*') if p.is_dir()]
            disc_subjects_by_disc[sub.name] = len(disc_subjects)
            print(f"  {sub.name}: {len(disc_subjects)} subjects")
            candidate_subject_dirs += disc_subjects

# dedupe & sort
print(f"\nBefore deduplication: {len(candidate_subject_dirs)} subjects")
candidate_subject_dirs = sorted({p.resolve(): p for p in candidate_subject_dirs}.values(), key=lambda x: x.name)
print(f"After deduplication: {len(candidate_subject_dirs)} subjects")

print("\nDisc breakdown:")
for disc_name, count in disc_subjects_by_disc.items():
    print(f"  {disc_name}: {count} subjects")

# Check for duplicates
from collections import Counter
subject_names = [p.name for p in candidate_subject_dirs]
duplicates = {name: count for name, count in Counter(subject_names).items() if count > 1}
if duplicates:
    print(f"\nDuplicate subjects found: {len(duplicates)}")
    print("First 5 duplicates:", list(duplicates.items())[:5])
else:
    print("\nNo duplicate subjects after deduplication")
