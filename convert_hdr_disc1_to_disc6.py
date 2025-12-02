"""
Convert HDR/IMG files to NII.GZ for disc1 through disc6
This script processes only the first 6 discs as requested.
"""
import sys
import traceback
from pathlib import Path

try:
    import nibabel as nib
except Exception as e:
    print("ERROR: nibabel not installed. Install with: pip install nibabel")
    raise

# === CONFIG ===
BASE = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")
DISCS_TO_PROCESS = ['disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'disc6']

print("=" * 80)
print("HDR to NII Converter - Processing disc1 through disc6")
print("=" * 80)
print(f"Base path: {BASE}\n")

if not BASE.exists():
    print("ERROR: base path does not exist. Check the path above.")
    sys.exit(1)

total_converted = 0
total_failed = 0
total_skipped = 0

for disc_name in DISCS_TO_PROCESS:
    disc_path = BASE / disc_name
    
    if not disc_path.exists():
        print(f"\n⚠️  {disc_name} not found, skipping...")
        continue
    
    print(f"\n{'='*80}")
    print(f"Processing {disc_name}")
    print(f"{'='*80}")
    
    subjects = sorted(list(disc_path.glob("OAS1_*_MR1")))
    print(f"Found {len(subjects)} subject directories in {disc_name}")
    
    disc_converted = 0
    disc_failed = 0
    disc_skipped = 0
    
    for subj in subjects:
        t88 = subj / "PROCESSED" / "MPRAGE" / "T88_111"
        if not t88.exists():
            continue
        
        hdrs = sorted(list(t88.glob("*.hdr")))
        imgs = sorted(list(t88.glob("*.img")))
        
        if len(hdrs) == 0:
            continue
        
        for hdr in hdrs:
            try:
                stem = hdr.stem
                expected_img = hdr.with_suffix('.img')
                out_path = hdr.with_suffix('.nii.gz')
                
                if out_path.exists():
                    disc_skipped += 1
                    continue
                
                if not expected_img.exists():
                    print(f"  ⚠️  {subj.name}: Missing .img for {hdr.name}")
                    continue
                
                print(f"  Converting {subj.name}/{hdr.name} → {out_path.name}...", end=' ')
                img = nib.load(str(hdr))
                nib.save(img, str(out_path))
                disc_converted += 1
                print("✓")
                
            except Exception as e:
                disc_failed += 1
                print(f"✗ FAILED: {e}")
    
    print(f"\n{disc_name} Summary:")
    print(f"  Converted: {disc_converted}")
    print(f"  Skipped (already exists): {disc_skipped}")
    print(f"  Failed: {disc_failed}")
    
    total_converted += disc_converted
    total_failed += disc_failed
    total_skipped += disc_skipped

print(f"\n{'='*80}")
print("FINAL SUMMARY - disc1 through disc6")
print(f"{'='*80}")
print(f"Total converted: {total_converted}")
print(f"Total skipped: {total_skipped}")
print(f"Total failed: {total_failed}")
print(f"\n✓ Conversion complete!")
