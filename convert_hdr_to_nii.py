# convert_hdr_to_nii.py
import sys
import traceback
from pathlib import Path

try:
    import nibabel as nib
except Exception as e:
    print("ERROR: nibabel not installed. Install with: pip install nibabel")
    raise

# === CONFIG - update if needed ===
# At the top, change:
BASE = Path(r"D:\Projects\AI-Projects\Alzimers\data\raw\mri")  # Root, not disc1
SUB_GLOB = "OAS1_*_MR1"

# Rest stays the same

print("Running convert_hdr_to_nii.py")
print("Base path:", BASE)
print()

if not BASE.exists():
    print("ERROR: base path does not exist. Check the path above.")
    sys.exit(1)

subjects = sorted(list(BASE.glob(SUB_GLOB)))
print(f"Found {len(subjects)} subject directories matching '{SUB_GLOB}'")

if len(subjects) == 0:
    print("No subjects found. Check whether your files are under a different subfolder (e.g. mri/ not mri/disc1).")
    sys.exit(0)

converted = 0
failed = 0

for subj in subjects:
    print("\nSubject:", subj.name)
    t88 = subj / "PROCESSED" / "MPRAGE" / "T88_111"
    if not t88.exists():
        print("  - T88_111 directory NOT FOUND:", t88)
        continue
    hdrs = sorted(list(t88.glob("*.hdr")))
    imgs = sorted(list(t88.glob("*.img")))
    print(f"  - Found {len(hdrs)} .hdr files and {len(imgs)} .img files in {t88}")

    if len(hdrs) == 0:
        continue

    for hdr in hdrs:
        try:
            # nibabel's Analyze loader expects the .img sibling to exist with same stem.
            stem = hdr.stem
            expected_img = hdr.with_suffix('.img')
            if not expected_img.exists():
                # if .img missing, warn but still try - nibabel may fail
                print(f"    * WARNING: matching .img not found for {hdr.name} (expected {expected_img.name})")

            out_path = hdr.with_suffix('.nii.gz')
            if out_path.exists():
                print(f"    - Skipping {hdr.name} (already converted -> {out_path.name})")
                continue

            print(f"    - Converting {hdr.name} -> {out_path.name} ...", end=' ')
            img = nib.load(str(hdr))   # will load .hdr/.img pair
            nib.save(img, str(out_path))
            converted += 1
            print("OK")
        except Exception as e:
            failed += 1
            print("FAILED")
            traceback.print_exc()

print()
print(f"Conversion complete. Converted: {converted}, Failed: {failed}")
print("If failures occurred, check the traceback above for details.")
