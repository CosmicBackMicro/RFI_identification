#!/usr/bin/env python3
"""按 cover 名分层随机打乱 SynthesizedDataset 的 train/val，并可按新清单复制构建新数据集。

首次运行（默认 dry-run）：
- 按 cover 前缀分层，在 train/val 内部随机打乱每个 cover 的样本
- 保持每个 cover 在 train/val 中的原始比例不变
- 输出计划摘要
- 保存打乱前/后的完整文件列表到 txt

提交运行（加 --commit）：
- 依据计划，复制构建一个新的数据集目录
- 不修改源数据集任何文件/文件夹
"""

from __future__ import annotations

import argparse
import shutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


IMAGE_EXT = ".fits"
MASK_EXT = ".png"
SPLITS = ("train", "val")


@dataclass(frozen=True)
class SplitPlan:
    """某个 split 的计划。"""

    source_split: str
    target_split: str
    count: int
    files: Tuple[Path, ...]


@dataclass(frozen=True)
class DatasetPlan:
    """整体重排计划。"""

    source_root: Path
    output_root: Path
    seed: int
    plans: Tuple[SplitPlan, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 cover 名分层随机打乱 SynthesizedDataset 的 train/val，并在 commit 时复制构建新数据集。"
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "源数据集目录（默认: 会在常见位置自动查找，例如 workspace 的 ``Datasets/SynthesizedDataset``，"
            "或用户主目录 ~/deRFI/Datasets/SynthesizedDataset，或历史路径 /home/bmcao/...；也可显式传入路径）"
        ),
    )
    parser.add_argument(
        "--output-name",
        default="SynthesizedDataset_Shuffled",
        help="输出数据集目录名（会创建在 source 的同级目录下）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="真正执行复制构建；不加则只输出计划和清单",
    )
    parser.add_argument(
        "--allow-mask-mismatch",
        action="store_true",
        help="若与图像同名的 mask 缺失，允许跳过该样本（默认严格要求 image/mask 成对存在）",
    )
    return parser.parse_args()


def find_default_source() -> Path | None:
    """在若干常见位置查找 SynthesizedDataset，找到第一个存在的路径返回，否则返回 None。

    候选（按优先级）：
    - 仓库根目录下的 `Datasets/SynthesizedDataset`（基于本文件父目录的父目录）
    - 用户主目录下的 `deRFI/Datasets/SynthesizedDataset`
    - 历史硬编码路径 `/home/bmcao/deRFI/Datasets/SynthesizedDataset`
    """
    candidates: List[Path] = []
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = None
    if repo_root is not None:
        candidates.append(repo_root / "Datasets" / "SynthesizedDataset")
    candidates.append(Path.home() / "deRFI" / "Datasets" / "SynthesizedDataset")
    candidates.append(Path("/home/bmcao/deRFI/Datasets/SynthesizedDataset"))

    for p in candidates:
        if p.exists():
            return p
    return None


def find_image_files(split_dir: Path) -> List[Path]:
    return sorted(p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() == IMAGE_EXT)


def image_to_mask_path(image_path: Path, mask_split_dir: Path) -> Path:
    return mask_split_dir / (image_path.stem + MASK_EXT)


def detect_cover_name(stem: str) -> str:
    """从文件 stem 中提取 cover 名：去掉末尾的 _block123。"""
    if "_block" in stem:
        prefix, suffix = stem.rsplit("_block", 1)
        if suffix.isdigit():
            return prefix
    # 兜底：若不符合 block 格式，整个 stem 视为一个 cover
    return stem


def build_cover_groups(image_files: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in image_files:
        cover = detect_cover_name(path.stem)
        groups.setdefault(cover, []).append(path)
    for files in groups.values():
        files.sort()
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def verify_pairs(image_files: Sequence[Path], mask_split_dir: Path) -> List[Path]:
    paired: List[Path] = []
    for img in image_files:
        mask = image_to_mask_path(img, mask_split_dir)
        if mask.exists():
            paired.append(img)
        else:
            raise FileNotFoundError(f"Missing mask for image: {img}")
    return paired


def collect_split(image_dir: Path, mask_dir: Path, allow_mask_mismatch: bool) -> Tuple[List[Path], List[Path]]:
    image_files = find_image_files(image_dir)
    if allow_mask_mismatch:
        paired_files = [p for p in image_files if image_to_mask_path(p, mask_dir).exists()]
    else:
        paired_files = verify_pairs(image_files, mask_dir)
    return paired_files, [image_to_mask_path(p, mask_dir) for p in paired_files]


def make_plan(source_root: Path, output_name: str, seed: int, allow_mask_mismatch: bool) -> Tuple[DatasetPlan, Dict[str, List[Path]]]:
    rng = random.Random(seed)
    output_root = source_root.parent / output_name

    source_image_root = source_root / "image"
    source_mask_root = source_root / "mask"
    if not source_image_root.exists():
        raise FileNotFoundError(f"Missing image root: {source_image_root}")
    if not source_mask_root.exists():
        raise FileNotFoundError(f"Missing mask root: {source_mask_root}")

    split_to_files: Dict[str, List[Path]] = {}
    for split in SPLITS:
        image_dir = source_image_root / split
        mask_dir = source_mask_root / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {mask_dir}")
        split_to_files[split], _ = collect_split(image_dir, mask_dir, allow_mask_mismatch)

    # 按 cover 分层：在 train/val 内部分别打乱，每个 cover 保持原 split 比例
    cover_buckets: Dict[str, Dict[str, List[Path]]] = {}
    for split, files in split_to_files.items():
        groups = build_cover_groups(files)
        for cover, cover_files in groups.items():
            cover_buckets.setdefault(cover, {"train": [], "val": []})[split].extend(cover_files)

    plans: List[SplitPlan] = []
    for cover in sorted(cover_buckets):
        train_files = cover_buckets[cover]["train"]
        val_files = cover_buckets[cover]["val"]
        all_files = train_files + val_files
        rng.shuffle(all_files)

        n_train = len(train_files)
        n_val = len(val_files)
        if len(all_files) != n_train + n_val:
            raise RuntimeError(f"Internal count mismatch for cover {cover}")

        new_train = tuple(sorted(all_files[:n_train], key=lambda p: p.name))
        new_val = tuple(sorted(all_files[n_train:n_train + n_val], key=lambda p: p.name))
        plans.append(SplitPlan("train", "train", n_train, new_train))
        plans.append(SplitPlan("val", "val", n_val, new_val))

    dataset_plan = DatasetPlan(source_root=source_root, output_root=output_root, seed=seed, plans=tuple(plans))
    return dataset_plan, split_to_files


def write_list(path: Path, title: str, files: Sequence[Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n")
        for p in files:
            f.write(str(p) + "\n")


def flatten_plan_files(plan: DatasetPlan) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"train": [], "val": []}
    for item in plan.plans:
        out[item.target_split].extend(str(p) for p in item.files)
    for split in out:
        out[split] = sorted(out[split])
    return out


def summarize_plan(plan: DatasetPlan, original: Dict[str, List[Path]]) -> None:
    print("\n=== Shuffle plan summary ===")
    print(f"Source root : {plan.source_root}")
    print(f"Output root : {plan.output_root}")
    print(f"Seed        : {plan.seed}")
    for split in SPLITS:
        original_count = len(original[split])
        new_count = len(flatten_plan_files(plan)[split])
        print(f"{split:5s}      : {original_count} files -> {new_count} planned entries")

    # 额外给出 cover 级别信息（便于你快速确认）
    cover_count = len({detect_cover_name(p.stem) for files in original.values() for p in files})
    print(f"Covers      : {cover_count}")
    print("============================\n")


def copy_dataset(plan: DatasetPlan) -> None:
    if plan.output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {plan.output_root}. Remove it first or choose another --output-name."
        )

    for split in SPLITS:
        (plan.output_root / "image" / split).mkdir(parents=True, exist_ok=False)
        (plan.output_root / "mask" / split).mkdir(parents=True, exist_ok=False)

    original_masks: Dict[Path, Path] = {}
    for split in SPLITS:
        src_image_dir = plan.source_root / "image" / split
        src_mask_dir = plan.source_root / "mask" / split
        for img in find_image_files(src_image_dir):
            original_masks[img] = image_to_mask_path(img, src_mask_dir)

    # 重新读取计划中每个 split 的样本，并复制到新目录
    for split in SPLITS:
        target_image_dir = plan.output_root / "image" / split
        target_mask_dir = plan.output_root / "mask" / split
        target_files = sorted(
            [p for p in flatten_plan_files(plan)[split]],
            key=lambda s: Path(s).name,
        )
        for src_img_str in target_files:
            src_img = Path(src_img_str)
            src_mask = original_masks.get(src_img)
            if src_mask is None or not src_mask.exists():
                raise FileNotFoundError(f"Missing mask during copy: {src_img}")
            dst_img = target_image_dir / src_img.name
            dst_mask = target_mask_dir / src_mask.name
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)


def main() -> int:
    args = parse_args()
    # 允许不传 --source：若未指定，则尝试 find_default_source() 的候选位置
    if args.source is None:
        found = find_default_source()
        if found is None:
            raise FileNotFoundError(
                "--source 未指定，且在常见位置未找到 SynthesizedDataset。请手动指定 --source。"
            )
        source_root = found.resolve()
        print(f"Auto-detected source dataset at: {source_root}")
    else:
        source_root = Path(args.source).expanduser().resolve()
    plan, original = make_plan(source_root, args.output_name, args.seed, args.allow_mask_mismatch)

    summarize_plan(plan, original)

    base_dir = source_root.parent
    before_txt = base_dir / f"{source_root.name}_shuffle_before.txt"
    after_txt = base_dir / f"{plan.output_root.name}_shuffle_after.txt"

    # 写入原始清单（按 split, 原始顺序）
    before_files = [p for split in SPLITS for p in original[split]]
    write_list(before_txt, "Original file list", before_files)

    # 写入计划清单：不要对文件进行全局排序，以便能看出哪些文件的 split 被改变
    after_files = [Path(p) for split in SPLITS for p in flatten_plan_files(plan)[split]]
    write_list(after_txt, "Planned shuffled file list", after_files)

    # 生成变动报告：哪些文件的 split 发生了变化（例如 train -> val）
    def report_changes(orig: Dict[str, List[Path]], plan: DatasetPlan, out_path: Path) -> None:
        orig_map: Dict[str, str] = {str(p): s for s in SPLITS for p in orig[s]}
        planned = flatten_plan_files(plan)
        changes: List[str] = []
        for s in SPLITS:
            for p in planned[s]:
                pstr = str(p)
                prev = orig_map.get(pstr)
                if prev is None:
                    changes.append(f"NEW\t{pstr}\t-> {s}")
                elif prev != s:
                    changes.append(f"MOVED\t{pstr}\t{prev} -> {s}")
        with out_path.open("w", encoding="utf-8") as f:
            f.write("Change report\n")
            f.write("=============\n")
            if not changes:
                f.write("No files changed split.\n")
            else:
                for line in changes:
                    f.write(line + "\n")

    changes_txt = base_dir / f"{plan.output_root.name}_shuffle_changes.txt"
    report_changes(original, plan, changes_txt)

    print(f"Saved original listing to: {before_txt}")
    print(f"Saved shuffled listing to: {after_txt}")

    # 更详细一点，方便你首次 dry-run 检查
    for split in SPLITS:
        print(f"{split} original count : {len(original[split])}")
        print(f"{split} planned count   : {len(flatten_plan_files(plan)[split])}")
    print()

    if not args.commit:
        print("\nDry-run only. Re-run with --commit to build the shuffled dataset by copying files.")
        return 0

    print("\nBuilding shuffled dataset by copying files...")
    copy_dataset(plan)
    print(f"Done. New dataset created at: {plan.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
