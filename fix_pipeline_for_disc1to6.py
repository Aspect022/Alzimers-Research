"""
Quick fix script to update phase1_data_pipeline.py to use disc1-disc6 only
"""
import re
from pathlib import Path

pipeline_file = Path(r"D:\Projects\AI-Projects\Alzimers\phase1_data_pipeline.py")
backup_file = Path(r"D:\Projects\AI-Projects\Alzimers\phase1_data_pipeline.py.backup")

# Read the file
with open(pipeline_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Create backup
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"✓ Backup created: {backup_file}")

# Find the section to replace
old_pattern = r"""# ========================================================================
# SECTION 4: Checking MRI Files \(search across discs\)
# ========================================================================
print\("\\n\[3/7\] Checking MRI Files"\)
print\("-" \* 80\)

root_mri_dir = Path\(config\['directories'\]\['data_raw'\]\) / 'mri'
candidate_subject_dirs = \[\]

if root_mri_dir\.exists\(\):
    # OAS1_\* directly under root
    candidate_subject_dirs \+= \[p for p in root_mri_dir\.glob\('OAS1_\*'\) if p\.is_dir\(\)\]
    # OAS1_\* under any disc\* subdir \(disc1\.\.disc5\)
    for sub in sorted\(root_mri_dir\.iterdir\(\)\):
        if sub\.is_dir\(\) and sub\.name\.lower\(\)\.startswith\('disc'\):
            candidate_subject_dirs \+= \[p for p in sub\.glob\('OAS1_\*'\) if p\.is_dir\(\)\]

# dedupe & sort
candidate_subject_dirs = sorted\(\{p\.resolve\(\): p for p in candidate_subject_dirs\}\.values\(\), key=lambda x: x\.name\)"""

new_code = """# ========================================================================
# SECTION 4: Checking MRI Files (USING disc1-disc6 ONLY)
# ========================================================================
print("\\n[3/7] Checking MRI Files (disc1-disc6)")
print("-" * 80)

root_mri_dir = Path(config['directories']['data_raw']) / 'mri'
candidate_subject_dirs = []
subject_dict = {}  # Track by name to avoid duplicates

# Process ONLY disc1 through disc6
DISCS_TO_USE = ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']

if root_mri_dir.exists():
    disc_subjects_count = {}
    
    for disc_name in DISCS_TO_USE:
        disc_path = root_mri_dir / disc_name
        if disc_path.exists() and disc_path.is_dir():
            disc_subjects = [p for p in disc_path.glob('OAS1_*') if p.is_dir()]
            disc_subjects_count[disc_name] = len(disc_subjects)
            
            for subject_path in disc_subjects:
                subject_name = subject_path.name
                # Only add if not already present (first occurrence wins)
                if subject_name not in subject_dict:
                    subject_dict[subject_name] = subject_path
    
    candidate_subject_dirs = sorted(subject_dict.values(), key=lambda x: x.name)
    
    print(f"\\n📊 Subject Discovery Summary (disc1-disc6):")
    for disc_name, count in sorted(disc_subjects_count.items()):
        print(f"  - {disc_name}: {count} subjects")
    print(f"  Total unique subjects: {len(candidate_subject_dirs)}")

# Final sort by subject name
candidate_subject_dirs = sorted(candidate_subject_dirs, key=lambda x: x.name)"""

# Replace
new_content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

if new_content == content:
    print("\n⚠️  Pattern not found! File might already be modified or have different format.")
    print("Please manually edit the file at line ~146-164")
else:
    # Save
    with open(pipeline_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"\n✓ File updated: {pipeline_file}")
    print("✓ Changed discovery logic to use disc1-disc6 only")
    print("\nYou can now run: python phase1_data_pipeline.py")
