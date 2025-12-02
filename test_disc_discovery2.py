"""Test script to debug disc discovery - check paths"""
from pathlib import Path
from collections import defaultdict

root_mri_dir = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")
candidate_subject_dirs = []

if root_mri_dir.exists():
    # OAS1_* directly under root
    root_subjects = [p for p in root_mri_dir.glob('OAS1_*') if p.is_dir()]
    print(f"Subjects directly under root: {len(root_subjects)}")
    candidate_subject_dirs += root_subjects
    
    # OAS1_* under any disc* subdir
    for sub in sorted(root_mri_dir.iterdir()):
        if sub.is_dir() and sub.name.lower().startswith('disc'):
            disc_subjects = [p for p in sub.glob('OAS1_*') if p.is_dir()]
            print(f"  {sub.name}: {len(disc_subjects)} subjects")
            candidate_subject_dirs += disc_subjects

print(f"\nBefore deduplication: {len(candidate_subject_dirs)} subjects")

# Check some sample paths
print("\nSample paths:")
subject_paths = defaultdict(list)
for p in candidate_subject_dirs[:5]:
    subject_paths[p.name].append(str(p))
    
for name, paths in subject_paths.items():
    print(f"{name}:")
    for path in paths:
        print(f"  {path}")

# Try resolving
print("\nChecking resolved paths for OAS1_0001_MR1:")
oas1_paths = [p for p in candidate_subject_dirs if p.name == 'OAS1_0001_MR1']
for p in oas1_paths:
    print(f"  Original: {p}")
    print(f"  Resolved: {p.resolve()}")
    print()
