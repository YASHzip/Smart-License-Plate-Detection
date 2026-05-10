"""
prepare_indian_dataset.py - Smart License Plate Detection
Fine-tuning Data Preparation

Merges an Indian number plate dataset (downloaded from Kaggle) with the
existing generic dataset in archive/ to create a combined dataset for
fine-tuning the YOLOv5 model.

Supported input annotation formats (auto-detected):
    - YOLO txt  (.txt files alongside images)
    - Pascal VOC XML (.xml files in a separate annotations/ folder)

Output structure (written to indian_dataset/):
    indian_dataset/
        images/
            train/   (80%)
            val/     (10%)
            test/    (10%)
        labels/
            train/
            val/
            test/

Usage:
    # Step 1: Download a Kaggle Indian plate dataset manually (see COMMANDS.txt)
    # Step 2: Run this script pointing at the downloaded folder:
    python prepare_indian_dataset.py --new-dataset path/to/downloaded/dataset

    # To only use the existing archive/ data (no new download):
    python prepare_indian_dataset.py --archive-only
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
ARCHIVE_DIR     = BASE_DIR / "archive"
OUTPUT_DIR      = BASE_DIR / "indian_dataset"
OUT_YAML        = BASE_DIR / "indian_data.yaml"

IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_RATIOS    = (0.80, 0.10, 0.10)   # train / val / test


# ── XML → YOLO conversion (Pascal VOC) ───────────────────────────────────────

def xml_to_yolo_line(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse one Pascal VOC XML file and return YOLO-format lines."""
    lines = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        print(f"  [WARN] Malformed XML: {xml_path.name} — skipped")
        return lines

    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        xc, yc, bw, bh = (
            max(0.0, min(1.0, xc)),
            max(0.0, min(1.0, yc)),
            max(0.0, min(1.0, bw)),
            max(0.0, min(1.0, bh)),
        )
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


# ── Pair collection ───────────────────────────────────────────────────────────

def collect_pairs_yolo(img_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find (image, label_txt) pairs where both files exist.
    label_dir may equal img_dir (labels stored alongside images).
    """
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTENSIONS:
            continue
        lbl = label_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            pairs.append((img, lbl))
    return pairs


def collect_pairs_xml(img_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find (image, annotation_xml) pairs for Pascal VOC layout.
    Tries both .png and .jpg for each .xml.
    """
    pairs = []
    for xml in sorted(ann_dir.glob("*.xml")):
        img = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = img_dir / (xml.stem + ext)
            if candidate.exists():
                img = candidate
                break
        if img:
            pairs.append((img, xml))
    return pairs


def collect_existing_archive() -> list[tuple[Path, Path, str]]:
    """
    Collect all (image, label_txt, fmt) triples from the existing archive/.
    Returns list of (img_path, lbl_path, "yolo").
    """
    triples = []
    for split in ["train", "validation", "test"]:
        img_dir = ARCHIVE_DIR / "images" / split
        lbl_dir = ARCHIVE_DIR / "labels" / split
        if not img_dir.exists():
            continue
        for img, lbl in collect_pairs_yolo(img_dir, lbl_dir):
            triples.append((img, lbl, "yolo"))
    print(f"  Archive: {len(triples)} labelled images found.")
    return triples


def collect_new_dataset(root: Path) -> list[tuple[Path, Path, str]]:
    """
    Auto-detect annotation format in a freshly downloaded dataset folder.
    Supports:
        - YOLO layout:  root/images/  +  root/labels/
        - YOLO flat:    root/*.jpg    +  root/*.txt
        - VOC layout:   root/images/  +  root/annotations/  (XML)
    Returns list of (img_path, ann_path, fmt) where fmt is "yolo" or "xml".
    """
    triples = []

    # --- YOLO with images/ and labels/ sub-folders ---
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if img_dir.exists() and lbl_dir.exists():
        # May have train/val/test sub-splits; walk all
        for sub_img in [img_dir] + [p for p in img_dir.iterdir() if p.is_dir()]:
            sub_lbl_name = sub_img.relative_to(img_dir)
            sub_lbl = lbl_dir / sub_lbl_name
            if not sub_lbl.exists():
                sub_lbl = lbl_dir   # fallback: flat labels dir
            for img, lbl in collect_pairs_yolo(sub_img, sub_lbl):
                triples.append((img, lbl, "yolo"))
        if triples:
            print(f"  New dataset (YOLO split): {len(triples)} labelled images.")
            return triples

    # --- Pascal VOC: images/ + annotations/ ---
    ann_dir = root / "annotations"
    if img_dir.exists() and ann_dir.exists():
        for img, xml in collect_pairs_xml(img_dir, ann_dir):
            triples.append((img, xml, "xml"))
        if triples:
            print(f"  New dataset (Pascal VOC): {len(triples)} labelled images.")
            return triples

    # --- Flat directory: images and labels mixed together ---
    flat_imgs  = [p for p in root.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    flat_txts  = {p.stem for p in root.glob("*.txt")}
    flat_pairs = [(img, root / (img.stem + ".txt"))
                  for img in flat_imgs if img.stem in flat_txts]
    if flat_pairs:
        triples = [(img, lbl, "yolo") for img, lbl in flat_pairs]
        print(f"  New dataset (flat YOLO): {len(triples)} labelled images.")
        return triples

    # --- Flat directory with XML annotations ---
    flat_xmls = {p.stem for p in root.glob("*.xml")}
    xml_pairs = []
    for img in flat_imgs:
        if img.stem in flat_xmls:
            xml_pairs.append((img, root / (img.stem + ".xml"), "xml"))
    if xml_pairs:
        print(f"  New dataset (flat Pascal VOC): {len(xml_pairs)} labelled images.")
        return xml_pairs

    print(f"  [WARN] No labelled images found in {root}. Check folder structure.")
    return []


# ── Splitting & writing ───────────────────────────────────────────────────────

def split_and_copy(triples: list, seed: int = 42):
    """
    Shuffle, split 80/10/10, convert XML→YOLO where needed, copy to output.
    """
    random.seed(seed)
    random.shuffle(triples)

    n       = len(triples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": triples[:n_train],
        "val":   triples[n_train : n_train + n_val],
        "test":  triples[n_train + n_val :],
    }

    counters = {}
    for split, items in splits.items():
        img_out = OUTPUT_DIR / "images" / split
        lbl_out = OUTPUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok = 0
        for img_src, ann_src, fmt in items:
            # Unique stem to avoid name collisions across datasets
            stem = img_src.stem
            dst_img = img_out / (stem + img_src.suffix)
            dst_lbl = lbl_out / (stem + ".txt")

            # Avoid overwriting if name already exists
            if dst_img.exists():
                stem    = stem + "_x"
                dst_img = img_out / (stem + img_src.suffix)
                dst_lbl = lbl_out / (stem + ".txt")

            # Copy image
            shutil.copy2(img_src, dst_img)

            # Write label
            if fmt == "yolo":
                shutil.copy2(ann_src, dst_lbl)
            else:  # XML → YOLO
                try:
                    with Image.open(img_src) as im:
                        w, h = im.size
                except Exception:
                    continue
                lines = xml_to_yolo_line(ann_src, w, h)
                if not lines:
                    continue
                dst_lbl.write_text("\n".join(lines) + "\n")

            ok += 1

        counters[split] = ok

    return counters


# ── YAML writer ───────────────────────────────────────────────────────────────

def write_yaml():
    content = f"""\
# indian_data.yaml — merged Indian + generic plate dataset
# Generated by prepare_indian_dataset.py

train: {(OUTPUT_DIR / 'images' / 'train').as_posix()}
val:   {(OUTPUT_DIR / 'images' / 'val').as_posix()}
test:  {(OUTPUT_DIR / 'images' / 'test').as_posix()}

nc: 1
names: ['license_plate']
"""
    OUT_YAML.write_text(content)
    print(f"\n  YAML written : {OUT_YAML}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare merged Indian + generic plate dataset for YOLOv5 fine-tuning"
    )
    parser.add_argument(
        "--new-dataset", type=str, default=None,
        help="Path to the downloaded Indian plate dataset root folder"
    )
    parser.add_argument(
        "--archive-only", action="store_true",
        help="Use only the existing archive/ data (skip new dataset)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Indian Dataset Preparation")
    print("=" * 60)

    # Clean previous output
    if OUTPUT_DIR.exists():
        print(f"\n[INFO] Removing old {OUTPUT_DIR.name}/ ...")
        shutil.rmtree(OUTPUT_DIR)

    all_triples = []

    # 1. Existing archive
    print("\n[1/2] Collecting existing archive/ data ...")
    all_triples += collect_existing_archive()

    # 2. New Indian dataset
    if not args.archive_only:
        if args.new_dataset is None:
            print("\n[ERROR] Please provide --new-dataset <path> or use --archive-only")
            print("        Download an Indian number plate dataset from Kaggle first.")
            print("        Recommended datasets (see COMMANDS.txt for direct links):")
            print("          • theiturhs/indian-vehicle-number-plate-dataset")
            print("          • nickyazdani/license-plate-detection")
            return
        new_root = Path(args.new_dataset)
        if not new_root.exists():
            print(f"\n[ERROR] Path not found: {new_root}")
            return
        print(f"\n[2/2] Collecting new dataset from: {new_root} ...")
        all_triples += collect_new_dataset(new_root)
    else:
        print("\n[2/2] --archive-only: skipping new dataset.")

    if not all_triples:
        print("\n[ERROR] No labelled images collected. Nothing to do.")
        return

    print(f"\n[INFO] Total labelled images collected : {len(all_triples)}")
    print(f"[INFO] Splitting 80 / 10 / 10 and copying to {OUTPUT_DIR.name}/ ...")

    counters = split_and_copy(all_triples, seed=args.seed)

    print(f"\n  Train : {counters['train']} images")
    print(f"  Val   : {counters['val']} images")
    print(f"  Test  : {counters['test']} images")

    write_yaml()

    print("\n" + "=" * 60)
    print("  Dataset ready. Next step — fine-tune:")
    print()
    print("  cd yolov5")
    print("  python train.py \\")
    print("    --img 640 --batch 16 --epochs 50 \\")
    print("    --data ../indian_data.yaml \\")
    print(r"    --weights runs\train\exp2\weights\best.pt \\")
    print("    --hyp ../hyp_indian.yaml \\")
    print("    --project runs/train --name indian_finetune \\")
    print("    --device 0 --cache")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
