#!/usr/bin/env python3
"""
安全地将多个数据集增量合并到 `Datasets/SynthesizedDataset`。

设计目标：
1. 默认 dry-run，只做扫描、抽样、冲突检查和 manifest 输出。
2. commit 阶段仅执行 manifest 中的动作。
3. 对 34 个真实观测 `Dataset_*`：
   - train 抽取至多 1000 个，不足则全取
   - val 抽取 train 数的 1/4，向下取整，不足则全取
4. `PointReinforced` 与 `SimulateDataset` 全量加入。
5. 保护 `SynthesizedDataset` 中现有 pulsar 数据（B0355 / B1929 / J195401 前缀）。
6. 用户已确认“前期不存在命名冲突”，因此若扫描到 Dataset 之间或与目标集的新增样本重名，直接报错中止。
7. 由于存储空间限制，commit 时对来源样本执行 move，而不是 copy。

注意：
- 该脚本不会删除 `SynthesizedDataset` 原有文件。
- commit 只会 move 计划中选中的来源样本（image 与对应 mask）。
- dry-run 建议先反复执行，确认 0 冲突后再 commit。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROTECTED_PREFIXES = (
    "B0355+54_20191110",
    "B1929+10_20210106",
    "J195401+292434_20210402",
)

DEFAULT_SYNTH_NAME = "SynthesizedDataset"
DEFAULT_MANIFEST_DIRNAME = "manifests"
DEFAULT_SEED = 20260308
REAL_DATASET_PREFIX = "Dataset_None_Specified"  # 修改此处以跳过真实观测数据集
FULL_MERGE_DATASETS = ()
SAMPLED_META_DATASETS = ("Dataset_PointEnhance",)  # 仅保留 PointEnhance
SAMPLED_META_LIMIT = 7000
SAMPLED_META_VAL_LIMIT = 7000 // 4  # 验证集对应抽取 1750 个样本


class MergePlanError(RuntimeError):
    """用于表示计划阶段发现的高风险错误。"""


@dataclass
class SamplePair:
    dataset_name: str
    split: str  # train / val
    image_src: str
    mask_src: str
    image_name: str
    mask_name: str
    target_image: str
    target_mask: str
    mode: str  # sampled / full


@dataclass
class DatasetPlanSummary:
    dataset_name: str
    mode: str
    available_train: int
    selected_train: int
    available_val: int
    selected_val: int


def list_files_sorted(folder: Path, suffix: str) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()])


def find_standard_mask_path(mask_dir: Path, image_path: Path) -> Path:
    return mask_dir / image_path.with_suffix(".png").name


def collect_standard_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = dataset_dir / "image" / split
    mask_dir = dataset_dir / "mask" / split
    image_files = list_files_sorted(image_dir, ".fits")
    pairs: List[Tuple[Path, Path]] = []
    missing_masks: List[str] = []

    for image_path in image_files:
        mask_path = find_standard_mask_path(mask_dir, image_path)
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
            continue
        pairs.append((image_path, mask_path))

    if missing_masks:
        preview = "\n".join(missing_masks[:20])
        extra = "" if len(missing_masks) <= 20 else f"\n... 另有 {len(missing_masks) - 20} 个缺失项"
        raise MergePlanError(
            f"数据集 `{dataset_dir.name}` 的 `{split}` 中存在缺失 mask，已中止。\n{preview}{extra}"
        )

    return pairs


def collect_simulate_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = dataset_dir / "image" / split
    mask_dir = dataset_dir / "mask" / split
    image_files = list_files_sorted(image_dir, ".fits")
    mask_files = [p for p in list_files_sorted(mask_dir, ".png") if not p.name.endswith("_plot.png")]

    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem: p for p in mask_files}

    extra_masks = sorted(set(mask_map) - set(image_map))
    if extra_masks:
        preview = "\n".join(extra_masks[:20])
        extra = "" if len(extra_masks) <= 20 else f"\n... 另有 {len(extra_masks) - 20} 个额外 mask"
        raise MergePlanError(
            f"数据集 `{dataset_dir.name}` 的 `{split}` 中存在无法匹配到 image 的 mask，已中止。\n{preview}{extra}"
        )

    pairs: List[Tuple[Path, Path]] = []
    missing_masks: List[str] = []
    for stem, image_path in image_map.items():
        mask_path = mask_map.get(stem)
        if mask_path is None:
            missing_masks.append(stem)
            continue
        pairs.append((image_path, mask_path))

    if missing_masks:
        preview = "\n".join(missing_masks[:20])
        extra = "" if len(missing_masks) <= 20 else f"\n... 另有 {len(missing_masks) - 20} 个缺失项"
        raise MergePlanError(
            f"数据集 `{dataset_dir.name}` 的 `{split}` 中存在缺失 mask，已中止。\n{preview}{extra}"
        )

    return sorted(pairs, key=lambda pair: pair[0].name)


def collect_pointreinforced_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = dataset_dir / "image" / split
    mask_dir = dataset_dir / "mask" / split
    if image_dir.is_dir() and mask_dir.is_dir():
        return collect_standard_pairs(dataset_dir, split)

    fits_dir = dataset_dir / "fits"
    masks_dir = dataset_dir / "masks"
    if fits_dir.is_dir() and masks_dir.is_dir():
        image_files = list_files_sorted(fits_dir, ".fits")
        mask_files = list_files_sorted(masks_dir, ".png")
        if image_files or mask_files:
            image_map = {p.stem: p for p in image_files}
            mask_map = {p.stem: p for p in mask_files}
            missing_masks = sorted(set(image_map) - set(mask_map))
            extra_masks = sorted(set(mask_map) - set(image_map))
            if missing_masks or extra_masks:
                details: List[str] = []
                if missing_masks:
                    details.append("缺失 mask:\n" + "\n".join(missing_masks[:20]))
                if extra_masks:
                    details.append("额外 mask:\n" + "\n".join(extra_masks[:20]))
                raise MergePlanError(
                    f"数据集 `{dataset_dir.name}` 的特殊目录结构存在配对异常，已中止。\n" + "\n".join(details)
                )
            return sorted([(image_map[k], mask_map[k]) for k in image_map], key=lambda pair: pair[0].name)

    return []


def collect_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    if dataset_dir.name == "SimulateDataset":
        return collect_simulate_pairs(dataset_dir, split)
    if dataset_dir.name == "PointReinforced":
        return collect_pointreinforced_pairs(dataset_dir, split)
    return collect_standard_pairs(dataset_dir, split)


def sample_pairs(pairs: Sequence[Tuple[Path, Path]], count: int, seed: int) -> List[Tuple[Path, Path]]:
    if count >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pairs)), count))
    return [pairs[i] for i in indices]


def choose_real_dataset_samples(dataset_dir: Path, seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], DatasetPlanSummary]:
    train_pairs = collect_pairs(dataset_dir, "train")
    val_pairs = collect_pairs(dataset_dir, "val")

    selected_train_n = min(1000, len(train_pairs))
    selected_train = sample_pairs(train_pairs, selected_train_n, seed)

    desired_val_n = selected_train_n // 4
    selected_val_n = min(desired_val_n, len(val_pairs))
    selected_val = sample_pairs(val_pairs, selected_val_n, seed + 1)

    summary = DatasetPlanSummary(
        dataset_name=dataset_dir.name,
        mode="sampled",
        available_train=len(train_pairs),
        selected_train=len(selected_train),
        available_val=len(val_pairs),
        selected_val=len(selected_val),
    )
    return selected_train, selected_val, summary


def choose_full_dataset_samples(dataset_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], DatasetPlanSummary]:
    train_pairs = collect_pairs(dataset_dir, "train")
    val_pairs = collect_pairs(dataset_dir, "val")
    summary = DatasetPlanSummary(
        dataset_name=dataset_dir.name,
        mode="full",
        available_train=len(train_pairs),
        selected_train=len(train_pairs),
        available_val=len(val_pairs),
        selected_val=len(val_pairs),
    )
    return train_pairs, val_pairs, summary


def collect_existing_names(synth_dir: Path) -> Tuple[set[str], set[str]]:
    image_names: set[str] = set()
    mask_names: set[str] = set()
    for split in ("train", "val"):
        image_dir = synth_dir / "image" / split
        mask_dir = synth_dir / "mask" / split
        image_names.update(p.name for p in list_files_sorted(image_dir, ".fits"))
        mask_names.update(p.name for p in list_files_sorted(mask_dir, ".png"))
    return image_names, mask_names


def validate_existing_protected_files(synth_dir: Path) -> None:
    for split in ("train", "val"):
        image_dir = synth_dir / "image" / split
        if not image_dir.is_dir():
            continue
        for file in image_dir.iterdir():
            if not file.is_file() or file.suffix.lower() != ".fits":
                continue
            # 这里不限制目标目录只存在受保护文件，只是确保现有 pulsar 数据可被识别。
            _ = any(file.name.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def build_sample_entries(
    dataset_name: str,
    split: str,
    pairs: Sequence[Tuple[Path, Path]],
    synth_dir: Path,
    mode: str,
) -> List[SamplePair]:
    entries: List[SamplePair] = []
    target_image_dir = synth_dir / "image" / split
    target_mask_dir = synth_dir / "mask" / split
    for image_path, mask_path in pairs:
        entries.append(
            SamplePair(
                dataset_name=dataset_name,
                split=split,
                image_src=str(image_path),
                mask_src=str(mask_path),
                image_name=image_path.name,
                mask_name=mask_path.name,
                target_image=str(target_image_dir / image_path.name),
                target_mask=str(target_mask_dir / mask_path.name),
                mode=mode,
            )
        )
    return entries


def detect_conflicts(entries: Sequence[SamplePair], existing_image_names: set[str], existing_mask_names: set[str]) -> None:
    seen_images: Dict[str, SamplePair] = {}
    seen_masks: Dict[str, SamplePair] = {}
    errors: List[str] = []

    for entry in entries:
        if entry.image_name in existing_image_names:
            errors.append(
                f"与目标集现有 image 重名: {entry.image_name} <- {entry.dataset_name}/{entry.split}"
            )
        if entry.mask_name in existing_mask_names:
            errors.append(
                f"与目标集现有 mask 重名: {entry.mask_name} <- {entry.dataset_name}/{entry.split}"
            )

        prev_img = seen_images.get(entry.image_name)
        if prev_img is not None:
            errors.append(
                "Dataset 间 image 命名冲突: "
                f"{entry.image_name} <- {prev_img.dataset_name}/{prev_img.split} 与 {entry.dataset_name}/{entry.split}"
            )
        else:
            seen_images[entry.image_name] = entry

        prev_mask = seen_masks.get(entry.mask_name)
        if prev_mask is not None:
            errors.append(
                "Dataset 间 mask 命名冲突: "
                f"{entry.mask_name} <- {prev_mask.dataset_name}/{prev_mask.split} 与 {entry.dataset_name}/{entry.split}"
            )
        else:
            seen_masks[entry.mask_name] = entry

    if errors:
        preview = "\n".join(errors[:50])
        extra = "" if len(errors) <= 50 else f"\n... 另有 {len(errors) - 50} 条冲突"
        raise MergePlanError("检测到命名冲突，已按约定中止：\n" + preview + extra)


def manifest_path_for(synth_dir: Path, name: str) -> Path:
    return synth_dir / DEFAULT_MANIFEST_DIRNAME / name


def plan_merge(datasets_root: Path, synth_name: str, seed: int) -> Tuple[List[SamplePair], List[DatasetPlanSummary], Dict[str, object]]:
    synth_dir = datasets_root / synth_name
    if not synth_dir.is_dir():
        raise MergePlanError(f"目标数据集不存在: {synth_dir}")

    validate_existing_protected_files(synth_dir)
    existing_image_names, existing_mask_names = collect_existing_names(synth_dir)

    all_entries: List[SamplePair] = []
    summaries: List[DatasetPlanSummary] = []

    top_level_dirs = sorted([p for p in datasets_root.iterdir() if p.is_dir()])
    real_dataset_dirs = [p for p in top_level_dirs if p.name.startswith(REAL_DATASET_PREFIX)]

    for idx, dataset_dir in enumerate(real_dataset_dirs):
        train_sel, val_sel, summary = choose_real_dataset_samples(dataset_dir, seed + idx * 17)
        summaries.append(summary)
        all_entries.extend(build_sample_entries(dataset_dir.name, "train", train_sel, synth_dir, "sampled"))
        all_entries.extend(build_sample_entries(dataset_dir.name, "val", val_sel, synth_dir, "sampled"))

    # 2. 处理高占比模拟/强化数据 (Dataset_PointReinforced, Dataset_Simulated)
    # 限制采样数量以防过拟合模拟分布
    for idx, dataset_name in enumerate(SAMPLED_META_DATASETS):
        dataset_dir = datasets_root / dataset_name
        if not dataset_dir.is_dir():
            print(f"警告: 预定义的元数据集不存在: {dataset_dir}")
            continue
            
        train_pairs = collect_pairs(dataset_dir, "train")
        val_pairs = collect_pairs(dataset_dir, "val")
        
        # 限制采样
        sel_train_n = min(SAMPLED_META_LIMIT, len(train_pairs))
        sel_train = sample_pairs(train_pairs, sel_train_n, seed + 1000 + idx)
        
        sel_val_n = min(SAMPLED_META_VAL_LIMIT, len(val_pairs))
        sel_val = sample_pairs(val_pairs, sel_val_n, seed + 2000 + idx)
        
        summary = DatasetPlanSummary(
            dataset_name=dataset_name,
            mode="sampled_meta",
            available_train=len(train_pairs),
            selected_train=len(sel_train),
            available_val=len(val_pairs),
            selected_val=len(sel_val)
        )
        summaries.append(summary)
        all_entries.extend(build_sample_entries(dataset_name, "train", sel_train, synth_dir, "sampled_meta"))
        all_entries.extend(build_sample_entries(dataset_name, "val", sel_val, synth_dir, "sampled_meta"))

    detect_conflicts(all_entries, existing_image_names, existing_mask_names)

    # 统计不同 downsamp 的样本分布 (仅针对真实观测)
    downsamp_stats: Dict[str, int] = {}
    for summary in summaries:
        if summary.mode == "sampled":
            # 提取 downsamp 数值，若无后缀默认为 downsamp1
            match = re.search(r"Downsamp(\d+)$", summary.dataset_name)
            ds_key = f"downsamp{match.group(1)}" if match else "downsamp1"
            downsamp_stats[ds_key] = downsamp_stats.get(ds_key, 0) + summary.selected_train

    report = {
        "datasets_root": str(datasets_root),
        "synthesized_dataset": str(synth_dir),
        "seed": seed,
        "real_dataset_count": len(real_dataset_dirs),
        "real_datasets": [p.name for p in real_dataset_dirs],
        "sampled_meta_datasets": list(SAMPLED_META_DATASETS),
        "downsamp_distribution": downsamp_stats,
        "selected_train_total": sum(1 for e in all_entries if e.split == "train"),
        "selected_val_total": sum(1 for e in all_entries if e.split == "val"),
    }
    return all_entries, summaries, report


def ensure_parent_dirs(entries: Iterable[SamplePair]) -> None:
    created: set[str] = set()
    for entry in entries:
        for path_str in (entry.target_image, entry.target_mask):
            parent = str(Path(path_str).parent)
            if parent not in created:
                Path(parent).mkdir(parents=True, exist_ok=True)
                created.add(parent)


def write_manifest(synth_dir: Path, entries: Sequence[SamplePair], summaries: Sequence[DatasetPlanSummary], report: Dict[str, object]) -> Path:
    manifest_dir = synth_dir / DEFAULT_MANIFEST_DIRNAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(synth_dir, "merge_into_synthesized_dataset_manifest.json")

    payload = {
        "report": report,
        "dataset_summaries": [asdict(summary) for summary in summaries],
        "entries": [asdict(entry) for entry in entries],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def execute_move(entries: Sequence[SamplePair]) -> None:
    ensure_parent_dirs(entries)
    for idx, entry in enumerate(entries, start=1):
        image_src = Path(entry.image_src)
        mask_src = Path(entry.mask_src)
        image_dst = Path(entry.target_image)
        mask_dst = Path(entry.target_mask)

        if not image_src.is_file():
            raise MergePlanError(f"commit 前源 image 已不存在: {image_src}")
        if not mask_src.is_file():
            raise MergePlanError(f"commit 前源 mask 已不存在: {mask_src}")
        if image_dst.exists():
            raise MergePlanError(f"commit 时目标 image 已存在，已中止: {image_dst}")
        if mask_dst.exists():
            raise MergePlanError(f"commit 时目标 mask 已存在，已中止: {mask_dst}")

        shutil.move(str(image_src), str(image_dst))
        shutil.move(str(mask_src), str(mask_dst))

        if idx % 500 == 0:
            print(f"  已移动 {idx} / {len(entries)} 对样本")


def print_summary(summaries: Sequence[DatasetPlanSummary], report: Dict[str, object], manifest_path: Optional[Path]) -> None:
    print("\n合并计划摘要：")
    print(f"  真实观测 Dataset 数量: {report['real_dataset_count']}")
    
    # 显式转换类型以消除 lint 警告
    from typing import cast
    ds_stats = cast(Dict[str, int], report.get("downsamp_distribution", {}))
    
    if ds_stats:
        print("  真实观测训练集 Downsamp 分布:")
        # 按 downsamp 数字排序输出
        for ds in sorted(ds_stats.keys(), key=lambda x: int(x.replace("downsamp", ""))):
            print(f"    - {ds:10s}: {ds_stats[ds]:5d} 样本")

    print(f"  计划新增总训练集:       {report['selected_train_total']}")
    print(f"  计划新增总验证集:       {report['selected_val_total']}")
    print("")
    for summary in summaries:
        print(
            f"  {summary.dataset_name:65s}"
            f" mode={summary.mode:7s}"
            f" avail_train={summary.available_train:5d}"
            f" take_train={summary.selected_train:5d}"
            f" avail_val={summary.available_val:5d}"
            f" take_val={summary.selected_val:5d}"
        )
    if manifest_path is not None:
        print(f"\nmanifest 已写出: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量合并到 SynthesizedDataset（默认 dry-run）")
    parser.add_argument("--datasets-root", default="/home/cbm/deRFI/Datasets", help="Datasets 根目录")
    parser.add_argument("--synth-name", default=DEFAULT_SYNTH_NAME, help="目标 SynthesizedDataset 名称")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="抽样随机种子")
    parser.add_argument("--commit", action="store_true", help="执行实际 move；默认仅 dry-run")
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="只生成 manifest，不执行 move（与默认 dry-run 类似，便于明确表达）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets_root = Path(args.datasets_root).resolve()
    synth_dir = datasets_root / args.synth_name

    try:
        entries, summaries, report = plan_merge(datasets_root, args.synth_name, args.seed)
        manifest_path = write_manifest(synth_dir, entries, summaries, report)
        print_summary(summaries, report, manifest_path)

        if args.commit:
            print("\n开始执行 move ...")
            execute_move(entries)
            print("move 完成。")
        else:
            print("\n当前为 dry-run，未执行任何 move。")

        return 0
    except MergePlanError as exc:
        print(f"\n[ERROR] {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - 兜底异常
        print(f"\n[UNEXPECTED ERROR] {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
