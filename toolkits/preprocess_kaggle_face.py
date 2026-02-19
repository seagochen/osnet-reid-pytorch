"""
Download and preprocess CASIA-WebFace dataset from Kaggle into a unified
format compatible with this project's face recognition training pipeline.

CASIA-WebFace structure (after extraction):
    CASIA-WebFace/
        <identity_id>/       # numeric folder name (e.g., 0000045)
            001.jpg           # face images for that identity
            002.jpg
            ...

Output structure:
    <output_dir>/
        images/               # All images as <person_id>_<seq>.jpg
        labels.csv            # img_path, person_id, camera_id

Usage:
    # Download from Kaggle and preprocess
    python toolkits/preprocess_kaggle_face.py --output-dir /path/to/output --download

    # Preprocess from an existing local copy
    python toolkits/preprocess_kaggle_face.py --output-dir /path/to/output \
        --dataset-dir /path/to/casia-webface

    # Limit identities (useful for quick experiments)
    python toolkits/preprocess_kaggle_face.py --output-dir /path/to/output \
        --download --max-ids 1000

    # Require minimum images per identity
    python toolkits/preprocess_kaggle_face.py --output-dir /path/to/output \
        --download --min-images 5
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def download_dataset(dest_dir: str) -> str:
    """Download CASIA-WebFace from Kaggle using kagglehub."""
    if os.path.exists(dest_dir):
        print(f"  Dataset already exists at: {dest_dir}, skipping download.")
        return dest_dir

    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub is required for downloading.")
        print("  pip install kagglehub")
        sys.exit(1)

    print("  Downloading CASIA-WebFace from Kaggle...")
    path = kagglehub.dataset_download("debarghamitraroy/casia-webface")
    shutil.move(path, dest_dir)
    print(f"  Downloaded to: {dest_dir}")
    return dest_dir


def find_image_root(dataset_dir: str) -> str:
    """
    Find the actual root directory containing identity subfolders.

    Kaggle datasets sometimes have extra nesting like:
        dataset_dir/CASIA-WebFace/CASIA-WebFace/<identity>/
    or just:
        dataset_dir/<identity>/

    We detect this by looking for the level where most children are
    numeric directories (identity folders).
    """
    candidates = [dataset_dir]

    # Check up to 3 levels of nesting
    for _ in range(3):
        next_candidates = []
        for cand in candidates:
            if not os.path.isdir(cand):
                continue

            children = os.listdir(cand)
            subdirs = [c for c in children if os.path.isdir(os.path.join(cand, c))]

            if not subdirs:
                continue

            # If most subdirs look like identity folders (numeric names), we found it
            numeric_count = sum(1 for d in subdirs if d.isdigit())
            if numeric_count > len(subdirs) * 0.5 and numeric_count > 10:
                print(f"  Found identity folders at: {cand} ({len(subdirs)} subdirs)")
                return cand

            # Otherwise go deeper
            for sd in subdirs:
                next_candidates.append(os.path.join(cand, sd))

        candidates = next_candidates

    # Fallback: return original dir
    print(f"  Warning: Could not auto-detect identity folder level, using: {dataset_dir}")
    return dataset_dir


def scan_identities(image_root: str, min_images: int = 1, max_ids: int = 0):
    """
    Scan identity subfolders and collect image paths.

    Returns:
        List of (identity_folder_name, list_of_image_paths)
    """
    entries = sorted(os.listdir(image_root))
    identities = []

    for folder_name in entries:
        folder_path = os.path.join(image_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        images = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])

        if len(images) < min_images:
            continue

        identities.append((folder_name, [os.path.join(folder_path, img) for img in images]))

        if max_ids > 0 and len(identities) >= max_ids:
            break

    return identities


def process_dataset(image_root: str, min_images: int = 1, max_ids: int = 0):
    """
    Process CASIA-WebFace into records for our training pipeline.

    Each identity folder becomes a person_id (sequential, starting from 1).
    camera_id is set to 0 (not applicable for face datasets).
    """
    identities = scan_identities(image_root, min_images=min_images, max_ids=max_ids)

    total_images = sum(len(imgs) for _, imgs in identities)
    print(f"  {len(identities)} identities, {total_images} images"
          f" (min {min_images} imgs/id"
          + (f", max {max_ids} ids" if max_ids > 0 else "") + ")")

    records = []
    for person_id_idx, (folder_name, image_paths) in enumerate(identities, start=1):
        person_id = str(person_id_idx)
        for seq, src_path in enumerate(image_paths, start=1):
            new_filename = f"{person_id}_{seq}.jpg"
            records.append({
                "src": src_path,
                "filename": new_filename,
                "person_id": person_id,
                "camera_id": "0",
            })

    return records


def copy_images(records: list, images_dir: str):
    """Copy images from source paths to the output images directory."""
    os.makedirs(images_dir, exist_ok=True)
    total = len(records)
    for i, rec in enumerate(records):
        dst = os.path.join(images_dir, rec["filename"])
        shutil.copy2(rec["src"], dst)
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"  Copied {i + 1}/{total} images...")


def write_csv(records: list, csv_path: str):
    """Write labels CSV in the project's expected format."""
    records.sort(key=lambda r: (int(r["person_id"]), r["filename"]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "person_id", "camera_id"])
        for rec in records:
            img_path = os.path.join("images", rec["filename"])
            writer.writerow([img_path, rec["person_id"], rec["camera_id"]])

    n_identities = len(set(r["person_id"] for r in records))
    print(f"  Wrote {len(records)} rows, {n_identities} identities -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess CASIA-WebFace for face recognition training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for organized dataset output")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Kaggle (requires kagglehub)")
    parser.add_argument("--dataset-dir", type=str, default="",
                        help="Path to existing CASIA-WebFace dataset")
    parser.add_argument("--min-images", type=int, default=1,
                        help="Minimum images per identity to include (default: 1)")
    parser.add_argument("--max-ids", type=int, default=0,
                        help="Maximum number of identities (0 = all, default: 0)")
    parser.add_argument("--csv-name", type=str, default="labels.csv",
                        help="Output CSV filename (default: labels.csv)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    # --- Locate dataset ---
    dataset_dir = args.dataset_dir
    if args.download:
        raw_dir = os.path.join(output_dir, "raw", "casia-webface")
        dataset_dir = download_dataset(raw_dir)
    elif not dataset_dir:
        print("Error: Provide --dataset-dir or use --download")
        sys.exit(1)

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # --- Find identity folders ---
    print("\nScanning dataset...")
    image_root = find_image_root(dataset_dir)

    # --- Process ---
    print("\nProcessing CASIA-WebFace...")
    records = process_dataset(image_root, min_images=args.min_images, max_ids=args.max_ids)

    if not records:
        print("\nNo images found. Check dataset structure.")
        sys.exit(1)

    # --- Copy images ---
    print(f"\nCopying {len(records)} images to {images_dir}...")
    copy_images(records, images_dir)

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, args.csv_name)
    print(f"\nWriting {csv_path}...")
    write_csv(records, csv_path)

    print(f"\nDone! Dataset ready at: {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {csv_path}")
    print(f"\nTo train:")
    print(f"  python scripts/train.py --config configs/face.yaml \\")
    print(f"      --data-root {output_dir} --csv {args.csv_name}")


if __name__ == "__main__":
    main()
