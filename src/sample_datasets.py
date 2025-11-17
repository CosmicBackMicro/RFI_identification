#!/usr/bin/env python3
"""
sample_datasets.py

扫描 Datasets 目录，统计每个数据集 image/train 中文件数量，按观测基名（去掉 _DownsampX 后缀）分组，
并按照用户给定的目标（orig=10000, d2=5000, d4=2500, d8=1250, d16=625）进行均匀分配抽样（dry-run 报告）。

使用示例：
  python3 scripts/sample_datasets.py --datasets-root Datasets --dry-run

可选 --commit 将写出 manifests/selected_train.txt（相对路径），否则仅打印报告。
"""

import os
import re
import argparse
import shutil
from collections import defaultdict, OrderedDict


def find_datasets(root):
    """返回 dataset_folder 列表（绝对或相对），只包含直接的子目录。"""
    names = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            names.append(entry)
    return names


def is_downsamp(name):
    m = re.search(r"_Downsamp(\d+)$", name)
    if m:
        return int(m.group(1))
    return 1


def base_obs_name(name):
    # 去掉尾部的 _DownsampX
    return re.sub(r"_Downsamp\d+$", "", name)


def count_train_images(dataset_root, dataset_name):
    train_dir = os.path.join(dataset_root, dataset_name, 'image', 'train')
    if not os.path.isdir(train_dir):
        return 0
    files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    return len(files)


def count_val_images(dataset_root, dataset_name):
    val_dir = os.path.join(dataset_root, dataset_name, 'image', 'val')
    if not os.path.isdir(val_dir):
        return 0
    files = [f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))]
    return len(files)


def allocate_evenly(candidates, target, avail_counts):
    """对候选观测列表 candidates 均匀分配 target 个样本（按每观测上限 avail_counts），返回 dict(obs->take)."""
    n = len(candidates)
    if n == 0:
        return {}, target
    per = target // n
    take = {}
    remaining = target
    for obs in candidates:
        cap = avail_counts.get(obs, 0)
        t = min(per, cap)
        take[obs] = t
        remaining -= t

    if remaining > 0:
        caps = sorted(candidates, key=lambda o: avail_counts.get(o, 0) - take.get(o,0), reverse=True)
        for obs in caps:
            if remaining <= 0:
                break
            space = avail_counts.get(obs, 0) - take.get(obs, 0)
            if space <= 0:
                continue
            add = min(space, remaining)
            take[obs] = take.get(obs,0) + add
            remaining -= add

    return take, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-root', default='Datasets', help='Datasets 根目录')
    parser.add_argument('--dry-run', action='store_true', default=True, help='只打印报告，不写文件')
    parser.add_argument('--commit', action='store_true', help='写出 manifests/selected_train.txt，并在提供 --new-name 时复制文件到新数据集')
    parser.add_argument('--new-name', type=str, help='在 --commit 时，将选中的样本复制到 Datasets/<new-name> 下的新数据集')
    # 移除 --force 与 --targets 功能：目录存在时直接退出，目标数量使用内置字典。
    args = parser.parse_args()

    root = args.datasets_root
    if not os.path.isdir(root):
        print(f"Datasets 根目录不存在: {root}")
        return

    ds_names = find_datasets(root)
    print(f"发现 {len(ds_names)} 个顶层数据集目录。\n")

    info = {}
    groups_by_base = defaultdict(dict)  # base_obs -> {factor: dataset_name}

    for name in ds_names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        factor = is_downsamp(name)
        base = base_obs_name(name)
        cnt_train = count_train_images(root, name)
        cnt_val = count_val_images(root, name)
        info[name] = {
            'factor': factor,
            'base': base,
            'train_count': cnt_train,
            'val_count': cnt_val,
            'rel_path': os.path.join(root, name)
        }
        groups_by_base[base][factor] = name

    orig_datasets = [n for n,v in info.items() if v['factor']==1]
    total_orig_train = sum(info[n]['train_count'] for n in orig_datasets)
    print(f"原始分辨率数据集数量 (factor=1): {len(orig_datasets)}，对应 train 文件总数: {total_orig_train}\n")

    targets = {1:10000, 2:5000, 4:2500, 8:1250, 16:625, 32:300, 64:150}

    selection = defaultdict(dict)  # factor -> {dataset_name: take_count}

    avail_by_factor = defaultdict(dict)  # factor -> base -> (dataset_name,avail)
    for base, dmap in groups_by_base.items():
        for factor, dname in dmap.items():
            avail = info[dname]['train_count']
            avail_by_factor[factor][base] = (dname, avail)

    for factor in sorted(targets.keys()):
        target = targets.get(factor, 0)
        candidates = list(avail_by_factor.get(factor, {}).keys())
        if not candidates:
            print(f"因子 {factor} 没有候选观测（可能缺少对应 Downsample 数据），需要 {target} 样本，缺失 {target} 个")
            continue

        avail_counts = {b: avail_by_factor[factor][b][1] for b in candidates}

        taken_map, remaining = allocate_evenly(candidates, target, avail_counts)

        for base, t in taken_map.items():
            dname = avail_by_factor[factor][base][0]
            selection[factor][dname] = t

        if remaining>0:
            print(f"因子 {factor} 目标 {target}，分配后仍差 {remaining} 个样本（请补齐对应数据）。")

    val_selection = defaultdict(dict)  # factor -> {dataset_name: val_take}
    for factor in sorted(selection.keys()):
        for dname, t in selection[factor].items():
            avail_val = info[dname].get('val_count', 0)
            v = min(avail_val, t // 4)
            if v > 0:
                val_selection[factor][dname] = v

    print('\n各数据集取样计划（train 部分，dry-run 报告）:')
    total_selected = 0
    total_selected_val = 0
    per_dataset_lines = []
    for factor in sorted(selection.keys()):
        for dname, t in sorted(selection[factor].items()):
            total_selected += t
            v = val_selection.get(factor, {}).get(dname, 0)
            total_selected_val += v
            per_dataset_lines.append((dname, info[dname]['base'], factor, info[dname]['train_count'], info[dname].get('val_count',0), t, v))

    for dname, base, factor, avail_tr, avail_val, take_tr, take_val in sorted(per_dataset_lines):
        print(f"{dname:70s}  factor={factor:2d}  avail_train={avail_tr:5d}  take_train={take_tr:5d}  avail_val={avail_val:5d}  take_val={take_val:5d}")

    print(f"\n总共选中样本数 (train): {total_selected}")
    print(f"总共选中样本数 (val):   {total_selected_val}")

    if args.commit:
        outdir = os.path.join(root, 'manifests')
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, 'selected_train.txt')
        outpath_val = os.path.join(outdir, 'selected_val.txt')
        with open(outpath, 'w') as fh, open(outpath_val, 'w') as fhv:
            for factor in sorted(selection.keys()):
                for dname, t in sorted(selection[factor].items()):
                    if t<=0:
                        continue
                    train_dir = os.path.join(root, dname, 'image', 'train')
                    files = sorted([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir,f))])
                    take_files = files[:t]
                    for f in take_files:
                        fh.write(os.path.join(root, dname, 'image', 'train', f) + '\n')
            for factor in sorted(val_selection.keys()):
                for dname, v in sorted(val_selection[factor].items()):
                    if v<=0:
                        continue
                    val_dir = os.path.join(root, dname, 'image', 'val')
                    files_v = sorted([f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir,f))])
                    take_files_v = files_v[:v]
                    for f in take_files_v:
                        fhv.write(os.path.join(root, dname, 'image', 'val', f) + '\n')
        print(f"已写出清单: {outpath} （train 文件路径，每行一条）")
        print(f"已写出清单: {outpath_val} （val   文件路径，每行一条）")

        if args.new_name:
            new_ds_dir = os.path.join(root, args.new_name)
            img_train_out = os.path.join(new_ds_dir, 'image', 'train')
            msk_train_out = os.path.join(new_ds_dir, 'mask', 'train')
            img_val_out = os.path.join(new_ds_dir, 'image', 'val')
            msk_val_out = os.path.join(new_ds_dir, 'mask', 'val')

            if os.path.exists(new_ds_dir):
                print(f"目标目录已存在: {new_ds_dir}，已退出（不覆盖现有内容）。")
                return

            os.makedirs(img_train_out, exist_ok=True)
            os.makedirs(msk_train_out, exist_ok=True)
            os.makedirs(img_val_out, exist_ok=True)
            os.makedirs(msk_val_out, exist_ok=True)

            copied_images = 0
            copied_masks = 0
            missing_masks = 0
            copied_images_val = 0
            copied_masks_val = 0
            missing_masks_val = 0
            basename_seen = set()

            def do_place(src, dst):
                shutil.copy2(src, dst)

            with open(outpath, 'r') as fh:
                print("\n开始复制训练集样本 ...")
                printed_train_datasets = set()
                for line in fh:
                    img_path = line.strip()
                    if not img_path:
                        continue
                    if not os.path.isfile(img_path):
                        print(f"跳过不存在的图像文件: {img_path}")
                        continue

                    base = os.path.basename(img_path)
                    if base in basename_seen:
                        print(f"检测到重复基名 {base}，已跳过该条")
                        continue
                    basename_seen.add(base)

                    # 复制图像
                    dst_img = os.path.join(img_train_out, base)
                    try:
                        do_place(img_path, dst_img)
                        copied_images += 1
                        parts = img_path.split(os.sep)
                        if 'image' in parts:
                            try:
                                idx_img = parts.index('image')
                                dname_part = parts[idx_img-1] if idx_img > 0 else 'UNKNOWN_DATASET'
                            except ValueError:
                                dname_part = 'UNKNOWN_DATASET'
                        else:
                            dname_part = 'UNKNOWN_DATASET'
                        if dname_part not in printed_train_datasets:
                            print(f"  -> 数据集 {dname_part} 开始复制训练样本")
                            printed_train_datasets.add(dname_part)
                        if copied_images % 200 == 0:
                            print(f"    已复制训练图像 {copied_images} 张")
                    except Exception as e:
                        print(f"复制图像失败: {img_path} -> {dst_img}: {e}")
                        continue

                    mask_path = img_path.replace(os.sep + 'image' + os.sep + 'train' + os.sep,
                                                 os.sep + 'mask' + os.sep + 'train' + os.sep)
                    mask_path = re.sub(r"\.fits$", ".png", mask_path)
                    dst_msk = os.path.join(msk_train_out, re.sub(r"\.fits$", ".png", base))
                    if os.path.isfile(mask_path):
                        try:
                            do_place(mask_path, dst_msk)
                            copied_masks += 1
                        except Exception as e:
                            print(f"复制掩码失败: {mask_path} -> {dst_msk}: {e}")
                    else:
                        print(f"缺少对应掩码，已跳过: {mask_path}")
                        missing_masks += 1

            # 复制 val 清单
            with open(outpath_val, 'r') as fhv:
                print("\n开始复制验证集样本 ...")
                printed_val_datasets = set()
                for line in fhv:
                    img_path = line.strip()
                    if not img_path:
                        continue
                    if not os.path.isfile(img_path):
                        print(f"跳过不存在的验证图像文件: {img_path}")
                        continue
                    base = os.path.basename(img_path)
                    dst_img = os.path.join(img_val_out, base)
                    try:
                        do_place(img_path, dst_img)
                        copied_images_val += 1
                        parts = img_path.split(os.sep)
                        if 'image' in parts:
                            try:
                                idx_img = parts.index('image')
                                dname_part = parts[idx_img-1] if idx_img > 0 else 'UNKNOWN_DATASET'
                            except ValueError:
                                dname_part = 'UNKNOWN_DATASET'
                        else:
                            dname_part = 'UNKNOWN_DATASET'
                        if dname_part not in printed_val_datasets:
                            print(f"  -> 数据集 {dname_part} 开始复制验证样本")
                            printed_val_datasets.add(dname_part)
                        if copied_images_val % 100 == 0:
                            print(f"    已复制验证图像 {copied_images_val} 张")
                    except Exception as e:
                        print(f"复制验证图像失败: {img_path} -> {dst_img}: {e}")
                        continue
                    mask_path = img_path.replace(os.sep + 'image' + os.sep + 'val' + os.sep,
                                                 os.sep + 'mask' + os.sep + 'val' + os.sep)
                    mask_path = re.sub(r"\.fits$", ".png", mask_path)
                    dst_msk = os.path.join(msk_val_out, re.sub(r"\.fits$", ".png", base))
                    if os.path.isfile(mask_path):
                        try:
                            do_place(mask_path, dst_msk)
                            copied_masks_val += 1
                        except Exception as e:
                            print(f"复制验证掩码失败: {mask_path} -> {dst_msk}: {e}")
                    else:
                        print(f"缺少对应验证掩码，已跳过: {mask_path}")
                        missing_masks_val += 1

            summary_path = os.path.join(new_ds_dir, 'merge_summary.txt')
            with open(summary_path, 'w') as sf:
                sf.write("# 合并摘要\n")
                sf.write("# 字段: dataset_name, base, factor, avail_train, take_train, avail_val, take_val\n")
                for dname, base, factor, avail_tr, avail_val, take_tr, take_val in sorted(per_dataset_lines):
                    sf.write(f"{dname}\t{base}\t{factor}\t{avail_tr}\t{take_tr}\t{avail_val}\t{take_val}\n")
                sf.write("\n# 选择方法: 按观测均匀分配 (even allocation)，每数据集按文件名排序取前K；验证集按 train:val=4:1 比例。\n")

            print(f"\n✅ 新数据集已构建: {new_ds_dir}")
            print(f"   复制 图像(train): {copied_images}，掩码(train): {copied_masks}，缺失掩码(train): {missing_masks}")
            print(f"   复制 图像(val):   {copied_images_val}，掩码(val):   {copied_masks_val}，缺失掩码(val):   {missing_masks_val}")
            print(f"   汇总文件: {summary_path}")
    else:
        print('\n这是 dry-run 模式；如需生成清单并写入文件，请加 --commit 参数。')


if __name__ == '__main__':
    main()
