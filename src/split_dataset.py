#!/usr/bin/env python3
"""
This script processes the 'output' folder containing FITS files and corresponding image/mask pairs.
It performs stratified sampling to split data into train/val sets, then renames the output folder.

Usage:
    python split_dataset.py --new_name <new_folder_name> [--val_ratio 0.1]

Example:
    python split_dataset.py --new_name dataset_v1 --val_ratio 0.1
"""

import os
import shutil
import argparse
import re
from pathlib import Path

def get_num(name):
    match = re.search(r'(?:block|mask_merged_)(\d+)', name)
    return int(match.group(1)) if match else 0

def main():
    parser = argparse.ArgumentParser(description="Split deRFI dataset into train/val sets.")
    parser.add_argument("--new_name", required=True, help="New name for the output folder.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of samples for validation (default: 0.2).")
    args = parser.parse_args()

    output_dir = Path("output")
    if not output_dir.exists():
        raise FileNotFoundError("Error: 'output' folder does not exist.")

    # Check FITS files count
    fits_files = sorted(output_dir.glob("*.fits"))
    if len(fits_files) < 10:
        raise ValueError(f"Error: Found only {len(fits_files)} FITS files in 'output'. At least 4000 required.")

    # Create image and mask folders
    image_dir = output_dir / "image"
    mask_dir = output_dir / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get FITS and PNG files from root, sorted by block number
    fits_files = sorted(output_dir.glob("*.fits"), key=lambda f: get_num(f.name))
    png_files = sorted(output_dir.glob("*.png"), key=lambda f: get_num(f.name))
    # if len(fits_files) != len(png_files):
    #     raise ValueError(f"Error: Number of FITS files ({len(fits_files)}) != PNG files ({len(png_files)}).")
    
    # Assume they are paired by sorted order

    total_samples = len(fits_files)
    val_interval = int(1 / args.val_ratio) if args.val_ratio > 0 else total_samples  # e.g., every 5th for 0.2 ratio

    # Create subfolders
    train_image_dir = image_dir / "train"
    val_image_dir = image_dir / "val"
    train_mask_dir = mask_dir / "train"
    val_mask_dir = mask_dir / "val"

    for d in [train_image_dir, val_image_dir, train_mask_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split and move files
    for i, (img_file, mask_file) in enumerate(zip(fits_files, png_files)):
        if i % val_interval == 0:
            # Move to val
            shutil.move(str(img_file), str(val_image_dir / img_file.name))
            shutil.move(str(mask_file), str(val_mask_dir / mask_file.name))
        else:
            # Move to train
            shutil.move(str(img_file), str(train_image_dir / img_file.name))
            shutil.move(str(mask_file), str(train_mask_dir / mask_file.name))

    # Move txt files to top level
    txt_files = list(output_dir.glob("*.txt"))
    for txt_file in txt_files:
        shutil.move(str(txt_file), str(output_dir / txt_file.name))

    # Count samples
    train_count = len(list(train_image_dir.glob('*.fits')))
    val_count = len(list(val_image_dir.glob('*.fits')))

    # Rename output folder
    new_dir = Path(args.new_name)
    if new_dir.exists():
        raise FileExistsError(f"Error: '{new_dir}' already exists.")
    output_dir.rename(new_dir)

    print(f"Dataset split complete:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train samples: {train_count}")
    print(f"  Val samples: {val_count}")
    print(f"  Output folder renamed to: {new_dir}")

if __name__ == "__main__":
    main()