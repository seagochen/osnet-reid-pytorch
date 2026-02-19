"""
Download and preprocess Kaggle ReID datasets (CUHK03 + Market-1501) into a
unified format compatible with this project's training pipeline.

Output structure:
    <output_dir>/
        images/                  # All images renamed as <person_id>_<seq>.jpg
        labels.csv               # img_path, person_id, camera_id

Datasets:
    - CUHK03 (images_labeled): person IDs 100001+
    - Market-1501 (bounding_box_train): person IDs 200001+

Usage:
    # Download from Kaggle and preprocess
    python toolkits/preprocess_kaggle_reid.py --output-dir /path/to/output --download

    # Preprocess from existing local copies
    python toolkits/preprocess_kaggle_reid.py --output-dir /path/to/output \
        --cuhk03-dir /path/to/cuhk03 --market1501-dir /path/to/market1501

    # Process only one dataset
    python toolkits/preprocess_kaggle_reid.py --output-dir /path/to/output \
        --download --datasets market1501
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path


def download_dataset(name: str, kaggle_slug: str, dest_dir: str) -> str:
    """Download a dataset from Kaggle using kagglehub."""
    if os.path.exists(dest_dir):
        print(f"  {name} already exists at: {dest_dir}, skipping download.")
        return dest_dir

    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub is required for downloading. Install with: pip install kagglehub")
        sys.exit(1)

    print(f"  Downloading {name} from Kaggle ({kaggle_slug})...")
    path = kagglehub.dataset_download(kaggle_slug)
    shutil.move(path, dest_dir)
    print(f"  Downloaded to: {dest_dir}")
    return dest_dir


def process_cuhk03(cuhk03_dir: str, output_dir: str, person_id_start: int = 100000):
    """
    Process CUHK03 dataset (images_labeled).

    Filename format: <pair>_<person>_<camera>_<seq>.png
    Example: 1_001_1_01.png -> person_id=100001, camera_id=1
    """
    labeled_dir = os.path.join(cuhk03_dir, "archive", "images_labeled")
    if not os.path.isdir(labeled_dir):
        print(f"  Warning: CUHK03 images_labeled not found at {labeled_dir}")
        # Try without archive/ subdirectory
        labeled_dir = os.path.join(cuhk03_dir, "images_labeled")
        if not os.path.isdir(labeled_dir):
            print(f"  Error: Cannot find images_labeled in {cuhk03_dir}")
            return []

    image_files = [f for f in os.listdir(labeled_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"  Found {len(image_files)} images in CUHK03 images_labeled")

    records = []
    for filename in image_files:
        parts = filename.split("_")
        if len(parts) < 4:
            print(f"  Skipping unexpected filename: {filename}")
            continue

        person_id = str(person_id_start + int(parts[1]))
        camera_id = parts[2]
        seq = parts[3].split(".")[0]

        new_filename = f"{person_id}_{camera_id}_{seq}.jpg"
        records.append({
            "src": os.path.join(labeled_dir, filename),
            "filename": new_filename,
            "person_id": person_id,
            "camera_id": camera_id,
        })

    print(f"  Processed {len(records)} CUHK03 images")
    return records


def process_market1501(market1501_dir: str, output_dir: str, person_id_start: int = 200000):
    """
    Process Market-1501 dataset (bounding_box_train).

    Filename format: <person>_c<cam>s<seq>_<frame>_<det>.jpg
    Example: 0002_c1s1_000451_01.jpg -> person_id=200002, camera_id=1
    """
    train_dir = os.path.join(market1501_dir, "bounding_box_train")
    if not os.path.isdir(train_dir):
        print(f"  Error: bounding_box_train not found at {train_dir}")
        return []

    image_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # Filter out junk images (person_id == -1 or 0000)
    image_files = [f for f in image_files if not f.startswith(("-1", "0000"))]
    print(f"  Found {len(image_files)} images in Market-1501 bounding_box_train")

    records = []
    cam_pattern = re.compile(r"c(\d+)")
    for filename in image_files:
        parts = filename.split("_")
        if len(parts) < 3:
            print(f"  Skipping unexpected filename: {filename}")
            continue

        orig_person_id = int(parts[0])
        person_id = str(person_id_start + orig_person_id)

        # Extract camera ID from c<N>s<N> segment
        cam_match = cam_pattern.search(parts[1])
        camera_id = cam_match.group(1) if cam_match else "0"

        frame_id = parts[2]
        det_id = parts[3].split(".")[0] if len(parts) > 3 else "0"

        new_filename = f"{person_id}_{camera_id}_{frame_id}_{det_id}.jpg"
        records.append({
            "src": os.path.join(train_dir, filename),
            "filename": new_filename,
            "person_id": person_id,
            "camera_id": camera_id,
        })

    print(f"  Processed {len(records)} Market-1501 images")
    return records


def copy_images(records: list, images_dir: str):
    """Copy images from source paths to the output images directory."""
    os.makedirs(images_dir, exist_ok=True)
    for i, rec in enumerate(records):
        dst = os.path.join(images_dir, rec["filename"])
        shutil.copy2(rec["src"], dst)
        if (i + 1) % 5000 == 0:
            print(f"  Copied {i + 1}/{len(records)} images...")
    print(f"  Copied {len(records)} images total")


def write_csv(records: list, csv_path: str):
    """Write labels CSV in the project's expected format."""
    # Sort by person_id, then filename for deterministic output
    records.sort(key=lambda r: (r["person_id"], r["filename"]))

    # Deduplicate by filename
    seen = set()
    unique = []
    for rec in records:
        if rec["filename"] not in seen:
            seen.add(rec["filename"])
            unique.append(rec)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "person_id", "camera_id"])
        for rec in unique:
            img_path = os.path.join("images", rec["filename"])
            writer.writerow([img_path, rec["person_id"], rec["camera_id"]])

    n_identities = len(set(r["person_id"] for r in unique))
    print(f"  Wrote {len(unique)} rows, {n_identities} identities -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess Kaggle ReID datasets for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for organized dataset output")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets from Kaggle (requires kagglehub)")
    parser.add_argument("--data-root", type=str, default="",
                        help="Root directory for downloaded datasets (default: <output-dir>/raw)")
    parser.add_argument("--cuhk03-dir", type=str, default="",
                        help="Path to existing CUHK03 dataset (overrides --data-root)")
    parser.add_argument("--market1501-dir", type=str, default="",
                        help="Path to existing Market-1501 dataset (overrides --data-root)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["cuhk03", "market1501"],
                        choices=["cuhk03", "market1501"],
                        help="Which datasets to process (default: both)")
    parser.add_argument("--csv-name", type=str, default="labels.csv",
                        help="Output CSV filename (default: labels.csv)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    data_root = os.path.abspath(args.data_root) if args.data_root else os.path.join(output_dir, "raw")
    images_dir = os.path.join(output_dir, "images")

    os.makedirs(output_dir, exist_ok=True)

    all_records = []

    # --- CUHK03 ---
    if "cuhk03" in args.datasets:
        print("\n[1/2] Processing CUHK03...")
        cuhk03_dir = args.cuhk03_dir or os.path.join(data_root, "cuhk03")

        if args.download:
            cuhk03_dir = download_dataset("CUHK03", "priyanagda/cuhk03", cuhk03_dir)

        if os.path.isdir(cuhk03_dir):
            records = process_cuhk03(cuhk03_dir, output_dir)
            all_records.extend(records)
        else:
            print(f"  CUHK03 not found at {cuhk03_dir}. Skipping.")

    # --- Market-1501 ---
    if "market1501" in args.datasets:
        print("\n[2/2] Processing Market-1501...")
        market1501_dir = args.market1501_dir or os.path.join(data_root, "ReID_Market1501")

        if args.download:
            market1501_dir = download_dataset("Market-1501", "rayiooo/reid_market-1501", market1501_dir)

        if os.path.isdir(market1501_dir):
            records = process_market1501(market1501_dir, output_dir)
            all_records.extend(records)
        else:
            print(f"  Market-1501 not found at {market1501_dir}. Skipping.")

    if not all_records:
        print("\nNo images found. Check your dataset paths or use --download.")
        sys.exit(1)

    # --- Copy images ---
    print(f"\nCopying {len(all_records)} images to {images_dir}...")
    copy_images(all_records, images_dir)

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, args.csv_name)
    print(f"\nWriting {csv_path}...")
    write_csv(all_records, csv_path)

    print(f"\nDone! Dataset ready at: {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {csv_path}")
    print(f"\nTo train:")
    print(f"  python scripts/train.py --config configs/reid.yaml \\")
    print(f"      --data-root {output_dir} --csv {args.csv_name}")


if __name__ == "__main__":
    main()
