import os
import glob
import random

def cleanup_point_enhance(target_train_count=1000, target_val_count=250):
    base_dir = "/home/cbm/deRFI/Datasets/SynthesizedDataset"
    
    # 定义待处理的目录
    paths = {
        "train": {
            "image": os.path.join(base_dir, "image/train"),
            "mask": os.path.join(base_dir, "mask/train")
        },
        "val": {
            "image": os.path.join(base_dir, "image/val"),
            "mask": os.path.join(base_dir, "mask/val")
        }
    }

    deletion_plan = {"train": [], "val": []}

    for split in ["train", "val"]:
        img_dir = paths[split]["image"]
        msk_dir = paths[split]["mask"]
        
        # 获取所有 PointEnhance_block 文件
        # 匹配模式: PointEnhance_block*.fits 和 PointEnhance_block*.png
        all_images = sorted(glob.glob(os.path.join(img_dir, "PointEnhance_block*.fits")))
        
        total_found = len(all_images)
        target_count = target_train_count if split == "train" else target_val_count
        
        if total_found <= target_count:
            print(f"[{split}] 找到 {total_found} 个文件，少于或等于目标 {target_count}，跳过。")
            continue

        # 随机选择保留的文件，其余删除
        random.seed(42) # 固定随机种子以便可重复
        to_keep = set(random.sample(all_images, target_count))
        to_delete_images = [f for f in all_images if f not in to_keep]

        print(f"[{split}] 找到 {total_found} 个，计划保留 {target_count} 个，计划删除 {len(to_delete_images)} 个。")

        # 检查配对并加入删除计划
        for img_path in to_delete_images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(msk_dir, f"{basename}.png")
            
            if os.path.exists(mask_path):
                deletion_plan[split].append((img_path, mask_path))
            else:
                print(f"[警告] 找不到配对的 Mask: {mask_path} (Image: {img_path})")
                # 即使找不到 mask，也建议删除该 image 以保持同步，但在计划中标记
                deletion_plan[split].append((img_path, None))

    # 列出清单 (每个 split 前 5 个例子)
    print("\n" + "="*50)
    print("删除清单示例 (前 5 个):")
    for split in ["train", "val"]:
        print(f"\n--- {split} ---")
        for i, (img, msk) in enumerate(deletion_plan[split][:5]):
            print(f"[{i+1}] Image: {os.path.basename(img)} | Mask: {os.path.basename(msk) if msk else 'MISSING!'}")
    
    total_del = len(deletion_plan["train"]) + len(deletion_plan["val"])
    print("\n" + "="*50)
    print(f"总计将删除 {total_del} 组文件 (Image + Mask)。")
    print("如果确认无误，请在脚本末尾调用 perform_deletion()。")
    
    return deletion_plan

def perform_deletion(plan):
    print("\n正在执行物理删除...")
    count = 0
    for split in ["train", "val"]:
        for img, msk in plan[split]:
            try:
                if os.path.exists(img):
                    os.remove(img)
                if msk and os.path.exists(msk):
                    os.remove(msk)
                count += 1
            except Exception as e:
                print(f"删除失败 {img}: {e}")
    print(f"成功删除 {count} 组文件。")

if __name__ == "__main__":
    # 第一步：预览
    plan = cleanup_point_enhance()
    
    # 如果你确定要删除了，可以取消下面这一行的注释：
    perform_deletion(plan)
