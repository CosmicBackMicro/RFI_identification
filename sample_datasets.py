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
    # 结构: <dataset_root>/<dataset_name>/image/train/*.fits
    train_dir = os.path.join(dataset_root, dataset_name, 'image', 'train')
    if not os.path.isdir(train_dir):
        return 0
    # 仅计数文件（不递归）
    files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    return len(files)


def allocate_evenly(candidates, target, avail_counts):
    """对候选观测列表 candidates 均匀分配 target 个样本（按每观测上限 avail_counts），返回 dict(obs->take)."""
    n = len(candidates)
    if n == 0:
        return {}, target
    per = target // n
    take = {}
    remaining = target
    # 第一轮基础分配
    for obs in candidates:
        cap = avail_counts.get(obs, 0)
        t = min(per, cap)
        take[obs] = t
        remaining -= t

    # 第二轮：按可用空间再分配剩余的样本
    if remaining > 0:
        # 按 avail space 降序填充
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
    parser.add_argument('--commit', action='store_true', help='写出 manifests/selected_train.txt')
    parser.add_argument('--targets', help='可选 JSON 文件指定 target 数量（未实现）')
    args = parser.parse_args()

    root = args.datasets_root
    if not os.path.isdir(root):
        print(f"Datasets 根目录不存在: {root}")
        return

    ds_names = find_datasets(root)
    print(f"发现 {len(ds_names)} 个顶层数据集目录（包含 Downsample 版本）。\n")

    # 收集信息: per dataset name -> factor, base_obs, train_count, path
    info = {}
    groups_by_base = defaultdict(dict)  # base_obs -> {factor: dataset_name}

    for name in ds_names:
        # skip scripts etc
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        factor = is_downsamp(name)
        base = base_obs_name(name)
        cnt = count_train_images(root, name)
        info[name] = {'factor': factor, 'base': base, 'train_count': cnt, 'rel_path': os.path.join(root, name)}
        groups_by_base[base][factor] = name

    # 汇总原始分辨率（factor==1）train 总数
    orig_datasets = [n for n,v in info.items() if v['factor']==1]
    total_orig_train = sum(info[n]['train_count'] for n in orig_datasets)
    print(f"原始分辨率数据集数量 (factor=1): {len(orig_datasets)}，对应 train 文件总数: {total_orig_train}\n")

    # 目标分配
    targets = {1:10000, 2:5000, 4:2500, 8:1250, 16:625}

    # 为了避免同一观测被不同因子重复使用，我们按观测 base 分配：
    # 对每一个 factor，候选观测是那些具有该 factor 的 base 并且尚未被分配。

    used_bases = set()
    selection = defaultdict(dict)  # factor -> {dataset_name: take_count}

    # 先构建每个 base 在对应 factor 下的可用 train_count（取 dataset 的 train_count）
    avail_by_factor = defaultdict(dict)  # factor -> base -> (dataset_name,avail)
    for base, dmap in groups_by_base.items():
        for factor, dname in dmap.items():
            avail = info[dname]['train_count']
            avail_by_factor[factor][base] = (dname, avail)

    # 按 factor 从小到大（1 -> 2 -> 4 -> 8 -> 16）分配
    for factor in [1,2,4,8,16]:
        target = targets.get(factor, 0)
        candidates = [b for b in avail_by_factor.get(factor, {}).keys() if b not in used_bases]
        # 如果没有候选，直接提示缺失
        if not candidates:
            print(f"因子 {factor} 没有候选观测（可能缺少对应 Downsample 数据），需要 {target} 样本，缺失 {target} 个")
            continue

        # avail_counts per base
        avail_counts = {b: avail_by_factor[factor][b][1] for b in candidates}

        # 均匀分配
        taken_map, remaining = allocate_evenly(candidates, target, avail_counts)

        # 记录并标记已用的 base（只要取到 >0 就标记为 used，避免同一观测被不同 factor 用到）
        for base, t in taken_map.items():
            dname = avail_by_factor[factor][base][0]
            selection[factor][dname] = t
            if t>0:
                used_bases.add(base)

        if remaining>0:
            print(f"因子 {factor} 目标 {target}，分配后仍差 {remaining} 个样本（请补齐对应数据）。")

    # 打印每个数据集将取样多少（只列出非零）
    print('\n各数据集取样计划（train 部分，dry-run 报告）:')
    total_selected = 0
    per_dataset_lines = []
    for factor in sorted(selection.keys()):
        for dname, t in sorted(selection[factor].items()):
            total_selected += t
            per_dataset_lines.append((dname, info[dname]['base'], factor, info[dname]['train_count'], t))

    # 按 dataset 名排序打印
    for dname, base, factor, avail, take in sorted(per_dataset_lines):
        print(f"{dname:70s}  factor={factor:2d}  avail_train={avail:5d}  take={take:5d}")

    print(f"\n总共选中样本数 (train): {total_selected}")

    # 如果 commit，写出 manifests/selected_train.txt
    if args.commit:
        outdir = os.path.join(root, 'manifests')
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, 'selected_train.txt')
        with open(outpath, 'w') as fh:
            for factor in sorted(selection.keys()):
                for dname, t in sorted(selection[factor].items()):
                    if t<=0:
                        continue
                    train_dir = os.path.join(root, dname, 'image', 'train')
                    files = sorted([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir,f))])
                    take_files = files[:t]
                    for f in take_files:
                        fh.write(os.path.join(root, dname, 'image', 'train', f) + '\n')
        print(f"已写出清单: {outpath} （train 文件路径，每行一条）")
    else:
        print('\n这是 dry-run 模式；如需生成清单并写入文件，请加 --commit 参数。')


if __name__ == '__main__':
    main()
