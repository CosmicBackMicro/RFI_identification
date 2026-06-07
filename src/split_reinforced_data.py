import os
import random
import shutil
import glob
from tqdm import tqdm

def split_dataset(source_dir, target_dir, val_count=2000):
    # Source paths
    fits_dir = os.path.join(source_dir, "fits")
    masks_dir = os.path.join(source_dir, "masks")
    
    # Target paths
    train_img_dir = os.path.join(target_dir, "image/train")
    val_img_dir = os.path.join(target_dir, "image/val")
    train_mask_dir = os.path.join(target_dir, "mask/train")
    val_mask_dir = os.path.join(target_dir, "mask/val")
    
    # Create target directories
    for d in [train_img_dir, val_img_dir, train_mask_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Get all FITS files
    fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))
    random.shuffle(fits_files)
    
    val_files = fits_files[:val_count]
    train_files = fits_files[val_count:]
    
    print(f"Moving {len(val_files)} files to validation set...")
    for fpath in tqdm(val_files):
        fname = os.path.basename(fpath)
        mask_name = fname.replace(".fits", ".png")
        mpath = os.path.join(masks_dir, mask_name)
        
        # Move image
        shutil.move(fpath, os.path.join(val_img_dir, fname))
        # Move mask
        if os.path.exists(mpath):
            shutil.move(mpath, os.path.join(val_mask_dir, mask_name))

    print(f"Moving {len(train_files)} files to training set...")
    for fpath in tqdm(train_files):
        fname = os.path.basename(fpath)
        mask_name = fname.replace(".fits", ".png")
        mpath = os.path.join(masks_dir, mask_name)
        
        # Move image
        shutil.move(fpath, os.path.join(train_img_dir, fname))
        # Move mask
        if os.path.exists(mpath):
            shutil.move(mpath, os.path.join(train_mask_dir, mask_name))

    print("Dataset split complete!")

if __name__ == "__main__":
    SOURCE = "/home/cbm/deRFI/Datasets/PointReinforced"
    TARGET = "/home/cbm/deRFI/Datasets/SynthesizedDataset"
    split_dataset(SOURCE, TARGET, val_count=2000)
