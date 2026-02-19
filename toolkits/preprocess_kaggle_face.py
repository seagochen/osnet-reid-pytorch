"""
Download and preprocess CASIA-WebFace dataset from Kaggle into a unified
format compatible with this project's face recognition training pipeline.

Supports two dataset formats:
  1. MXNet RecordIO (.rec + .idx) — the Kaggle CASIA-WebFace format
  2. Image folders (identity_id/001.jpg) — generic face dataset layout

Output structure:
    <output_dir>/
        images/               # All images as <person_id>_<seq>.jpg
        labels.csv            # img_path, person_id, camera_id

Usage:
    # Download from Kaggle and preprocess
    python toolkits/preprocess_kaggle_face.py --dataset-dir ~/datasets/face_data --download

    # Re-run (auto-detects raw data from previous download)
    python toolkits/preprocess_kaggle_face.py --dataset-dir ~/datasets/face_data

    # Quick experiment with fewer identities
    python toolkits/preprocess_kaggle_face.py --dataset-dir ~/datasets/face_data --max-ids 1000
"""

import argparse
import collections
import csv
import os
import shutil
import struct
import sys

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
RECORDIO_MAGIC = 0xced7230a
JPEG_MAGIC = b'\xff\xd8\xff'
MIN_IMAGE_SIZE = 100  # bytes — anything smaller is metadata, not a real image


# ============================================================
# MXNet RecordIO reader (pure Python, no mxnet dependency)
# ============================================================

def find_recordio_files(dataset_dir: str):
    """
    Search for train.rec + train.idx in the dataset directory tree.
    Returns (rec_path, idx_path) or (None, None) if not found.
    """
    for root, dirs, files in os.walk(dataset_dir):
        files_lower = {f.lower(): f for f in files}
        if 'train.rec' in files_lower and 'train.idx' in files_lower:
            rec_path = os.path.join(root, files_lower['train.rec'])
            idx_path = os.path.join(root, files_lower['train.idx'])
            return rec_path, idx_path
    return None, None


def read_idx_file(idx_path: str):
    """Parse .idx file -> dict of {record_index: byte_offset}."""
    offsets = {}
    with open(idx_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                offsets[int(parts[0])] = int(parts[1])
    return offsets


def read_record(f, offset: int):
    """
    Read a single MXNet RecordIO record at the given byte offset.
    Returns the raw data bytes, or None on failure.

    RecordIO format per record:
      4 bytes: uint32 magic (0xced7230a)
      4 bytes: uint32 (cflag << 29 | length)
      <length> bytes: data
      padding to 4-byte boundary
    """
    f.seek(offset)
    buf = f.read(4)
    if len(buf) < 4:
        return None
    magic = struct.unpack('<I', buf)[0]
    if magic != RECORDIO_MAGIC:
        return None

    lrecord = struct.unpack('<I', f.read(4))[0]
    cflag = (lrecord >> 29) & 7
    length = lrecord & ((1 << 29) - 1)
    data = f.read(length)

    # Handle multi-part records
    if cflag == 1:  # start of multi-part
        while True:
            f.read((4 - (length % 4)) % 4)  # skip padding
            f.read(4)  # skip magic
            lr2 = struct.unpack('<I', f.read(4))[0]
            cf2 = (lr2 >> 29) & 7
            len2 = lr2 & ((1 << 29) - 1)
            data += f.read(len2)
            if cf2 == 3:  # end of multi-part
                break

    return data


def parse_irheader(data: bytes):
    """
    Parse InsightFace IRHeader from record data.
    Header: flag(uint32) + label(float32) + id(uint64) + id2(uint64) = 24 bytes
    Returns (label, image_bytes).
    """
    if len(data) < 24:
        return None, None
    _flag, label = struct.unpack('<If', data[:8])
    return int(label), data[24:]


def process_recordio(rec_path: str, idx_path: str, images_dir: str,
                     min_images: int = 1, max_ids: int = 0):
    """
    Extract images from MXNet RecordIO and produce training records.

    Returns list of record dicts with keys: filename, person_id, camera_id.
    Images are written directly to images_dir (no intermediate copy needed).
    """
    offsets = read_idx_file(idx_path)
    print(f"  RecordIO: {len(offsets)} records in index")

    # First pass: collect labels per record index (skip index 0 = metadata)
    print("  Scanning labels...")
    label_to_indices = collections.defaultdict(list)
    with open(rec_path, 'rb') as f:
        for idx in sorted(offsets.keys()):
            if idx == 0:  # metadata header
                continue
            data = read_record(f, offsets[idx])
            if data is None:
                continue
            label, img_bytes = parse_irheader(data)
            if (label is None or img_bytes is None
                    or len(img_bytes) < MIN_IMAGE_SIZE
                    or not img_bytes[:3].startswith(JPEG_MAGIC)):
                continue
            label_to_indices[label].append(idx)

    print(f"  Found {len(label_to_indices)} identities, "
          f"{sum(len(v) for v in label_to_indices.values())} images")

    # Filter by min_images
    valid_labels = sorted([
        lbl for lbl, indices in label_to_indices.items()
        if len(indices) >= min_images
    ])
    if max_ids > 0:
        valid_labels = valid_labels[:max_ids]

    total_images = sum(len(label_to_indices[lbl]) for lbl in valid_labels)
    print(f"  After filtering: {len(valid_labels)} identities, {total_images} images"
          f" (min {min_images} imgs/id"
          + (f", max {max_ids} ids" if max_ids > 0 else "") + ")")

    # Second pass: extract images
    os.makedirs(images_dir, exist_ok=True)
    records = []
    count = 0

    with open(rec_path, 'rb') as f:
        for person_id, orig_label in enumerate(valid_labels, start=1):
            indices = label_to_indices[orig_label]
            for seq, idx in enumerate(sorted(indices), start=1):
                data = read_record(f, offsets[idx])
                if data is None:
                    continue
                _, img_bytes = parse_irheader(data)
                if (img_bytes is None or len(img_bytes) < MIN_IMAGE_SIZE
                        or not img_bytes[:3].startswith(JPEG_MAGIC)):
                    continue

                filename = f"{person_id}_{seq}.jpg"
                img_path = os.path.join(images_dir, filename)
                with open(img_path, 'wb') as img_f:
                    img_f.write(img_bytes)

                records.append({
                    "filename": filename,
                    "person_id": str(person_id),
                    "camera_id": "0",
                })

                count += 1
                if count % 50000 == 0:
                    print(f"  Extracted {count}/{total_images} images...")

    print(f"  Extracted {count} images total")
    return records


# ============================================================
# Folder-based reader (identity_id/001.jpg)
# ============================================================

def find_image_root(dataset_dir: str):
    """
    Find the directory containing identity subfolders (numeric dirs with images).
    Searches up to 3 levels deep.
    """
    candidates = [dataset_dir]
    for _ in range(3):
        next_candidates = []
        for cand in candidates:
            if not os.path.isdir(cand):
                continue
            children = os.listdir(cand)
            subdirs = [c for c in children if os.path.isdir(os.path.join(cand, c))]
            if not subdirs:
                continue
            numeric_count = sum(1 for d in subdirs if d.isdigit())
            if numeric_count > len(subdirs) * 0.5 and numeric_count > 10:
                return cand
            for sd in subdirs:
                next_candidates.append(os.path.join(cand, sd))
        candidates = next_candidates
    return None


def process_folders(image_root: str, images_dir: str,
                    min_images: int = 1, max_ids: int = 0):
    """Process identity-folder structured dataset."""
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

    total_images = sum(len(imgs) for _, imgs in identities)
    print(f"  {len(identities)} identities, {total_images} images")

    os.makedirs(images_dir, exist_ok=True)
    records = []
    count = 0
    for person_id, (_, image_paths) in enumerate(identities, start=1):
        for seq, src_path in enumerate(image_paths, start=1):
            filename = f"{person_id}_{seq}.jpg"
            shutil.copy2(src_path, os.path.join(images_dir, filename))
            records.append({
                "filename": filename,
                "person_id": str(person_id),
                "camera_id": "0",
            })
            count += 1
            if count % 10000 == 0:
                print(f"  Copied {count}/{total_images} images...")

    print(f"  Processed {count} images total")
    return records


# ============================================================
# Common
# ============================================================

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
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Dataset directory (raw data, images, and labels.csv)")
    parser.add_argument("--download", action="store_true",
                        help="Download from Kaggle if not already present")
    parser.add_argument("--min-images", type=int, default=1,
                        help="Minimum images per identity (default: 1)")
    parser.add_argument("--max-ids", type=int, default=0,
                        help="Max identities, 0 = all (default: 0)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.dataset_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    # --- Download if requested ---
    if args.download:
        download_dataset(os.path.join(output_dir, "raw", "casia-webface"))

    # --- Detect format and process (search entire dataset_dir) ---
    rec_path, idx_path = find_recordio_files(output_dir)

    if rec_path and idx_path:
        print(f"\nDetected MXNet RecordIO format:")
        print(f"  rec: {rec_path}")
        print(f"  idx: {idx_path}")
        records = process_recordio(
            rec_path, idx_path, images_dir,
            min_images=args.min_images, max_ids=args.max_ids,
        )
    else:
        image_root = find_image_root(output_dir)
        if image_root is None:
            print(f"\nError: No dataset found in {output_dir}")
            print(f"  Use --download to fetch from Kaggle:")
            print(f"  python toolkits/preprocess_kaggle_face.py --dataset-dir {args.dataset_dir} --download")
            sys.exit(1)
        print(f"\nFound identity folders at: {image_root}")
        records = process_folders(
            image_root, images_dir,
            min_images=args.min_images, max_ids=args.max_ids,
        )

    if not records:
        print("\nNo images found. Check dataset structure.")
        sys.exit(1)

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, "labels.csv")
    print(f"\nWriting {csv_path}...")
    write_csv(records, csv_path)

    print(f"\nDone! Dataset ready at: {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {csv_path}")
    print(f"\nTo train:")
    print(f"  python scripts/train.py --config configs/face.yaml \\")
    print(f"      --data-root {output_dir} --csv labels.csv")


if __name__ == "__main__":
    main()
