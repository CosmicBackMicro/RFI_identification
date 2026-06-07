#!/usr/bin/env python3
"""src/EvaluateRFI.py

Evaluate segmentation masks against ground truth.

This script supports two evaluation modes:

1) multiclass (legacy): per-class IoU/Precision/Recall/F1 from a global multi-class confusion matrix.
2) binary_flat: flatten masks to binary for fair comparison with single-class methods (e.g. AOFlagger).

Binary flattening rule (as requested):

- Positive (RFI)  : any non-zero pixel EXCEPT Pulsar(8)
- Negative        : 0 (Background)
- Ignored pixels  : Pulsar(8) in either GT or prediction (excluded from metrics)

Metrics reported in binary_flat mode: Accuracy, Precision, Recall, F1 and the 2x2 confusion matrix.
"""
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import time
from tqdm import tqdm
import pandas as pd
from typing import Optional
import multiprocessing as mp
from functools import partial

# RFI Class Definitions
CLASS_LABELS = {
    0: 'Background',
    1: 'Horizontal',
    2: 'Vertical',
    6: 'Point',
    7: 'Block',
    8: 'Pulsar'
}
LABELS_LIST = sorted(CLASS_LABELS.keys()) # [0, 1, 2, 6, 7, 8]
NUM_CLASSES = len(LABELS_LIST)
LABEL_TO_INDEX = {lbl: idx for idx, lbl in enumerate(LABELS_LIST)}

# AI Model Label Mapping (Continuous 0-5 to Sparse GT Labels)
# 0->0, 1->1, 2->2, 3->6(Point), 4->7(Block), 5->8(Pulsar)
AI_LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 6,
    4: 7,
    5: 8
}

def remap_ai_labels(y: np.ndarray):
    """Remap continuous AI labels [0..5] and their remapped equivalents to sparse GT labels [0,1,2,6,7,8]."""
    out = np.zeros_like(y)
    # 确保 3 (原始输出) 和 6 (已重映射输出) 都被正确识别为 Point
    target_6_mask = (y == 3) | (y == 6)
    target_7_mask = (y == 4) | (y == 7)
    target_8_mask = (y == 5) | (y == 8)
    
    out[y == 1] = 1
    out[y == 2] = 2
    out[target_6_mask] = 6
    out[target_7_mask] = 7
    out[target_8_mask] = 8
    return out


def parse_block_index(filename: str):
    """Extract block index as int from filenames like:

    - simulation_psrfits_block0000.png
    - simulation_psrfits_block0.png
    """
    basename = os.path.basename(filename)
    m = re.search(r"block(\d+)", basename)
    return int(m.group(1)) if m else None


def find_gt_path(gt_dir: str, file_idx: int):
    """Find the matching GT mask path.

    Supports either:
    - mask_{idx}.png (legacy)
    - simulation_psrfits_block{idx}.png / block{idx:04d}.png (simulation_and_compare)
    """
    # legacy
    p1 = os.path.join(gt_dir, f"mask_{file_idx}.png")
    if os.path.exists(p1):
        return p1

    # simulation_and_compare variants
    candidates = [
        os.path.join(gt_dir, f"simulation_psrfits_block{file_idx}.png"),
        os.path.join(gt_dir, f"simulation_psrfits_block{file_idx:04d}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def detect_pred_glob(pred_dir: str) -> str:
    """Auto-detect a reasonable glob pattern for prediction mask files in pred_dir.

    Tries a few common filename styles and returns a glob string. Falls back to '*.png'.
    """
    try:
        fns = [fn for fn in os.listdir(pred_dir) if fn.lower().endswith('.png')]
    except Exception:
        return '*.png'

    # Heuristics in order of preference
    for fn in fns:
        if 'simulation_psrfits_block' in fn:
            return 'simulation_psrfits_block*.png'
    for fn in fns:
        if fn.startswith('mask_'):
            return 'mask_*.png'
    for fn in fns:
        if 'block' in fn:
            return '*block*.png'
    # fallback to any png
    if fns:
        return '*.png'
    return '*.png'

def load_mask(path):
    """Load mask image and flatten it."""
    try:
        img = Image.open(path)
        arr = np.array(img).astype(np.int32)
        return arr.flatten()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def process_single_pair(pred_path, gt_dir, is_aoflagger=False, eval_mode='multiclass'):
    """
    Core function for multiprocessing: Load a single GT/Pred mask pair and return CM.
    """
    idx = parse_block_index(pred_path)
    if idx is None:
        return None, "no_index"

    gt_path = find_gt_path(gt_dir, idx)
    if not gt_path:
        return None, "missing_gt"

    y_true = load_mask(gt_path)
    y_pred = load_mask(pred_path)
    
    if y_true is None or y_pred is None:
        return None, "load_error"
    
    # AI model label remapping
    if not is_aoflagger:
        y_pred = remap_ai_labels(y_pred)

    if y_true.size != y_pred.size:
        return None, f"size_mismatch_{idx}"

    if eval_mode == 'binary_flat':
        # Rule: ignore Pulsar(8) and Block(7) as per user request in binary mode
        y_true_bin, y_pred_bin = flatten_to_binary_ignore_pulsar(y_true, y_pred, pulsar_label=8, block_label=7)
        if y_true_bin.size == 0:
            return None, "ignored"
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        m_i = metrics_from_binary_cm(cm)
        return (idx, cm, m_i), "success"
    else:
        # Multiclass: Keep only defined labels, ignore Pulsar(8)
        mask_valid = np.isin(y_true, LABELS_LIST) & np.isin(y_pred, LABELS_LIST)
        y_true_v = y_true[mask_valid]
        y_pred_v = y_pred[mask_valid]
        
        if y_true_v.size == 0:
            return None, "ignored"
        
        cm = confusion_matrix(y_true_v, y_pred_v, labels=LABELS_LIST)
        return cm, "success"

def flatten_to_binary_ignore_pulsar(y_true: np.ndarray, y_pred: np.ndarray, pulsar_label: int = 8, block_label: int = 7):
    """Flatten multiclass labels to binary and ignore Pulsar.
    Note: Block (7) is now included in RFI as requested.

    Returns:
        y_true_bin, y_pred_bin as uint8 arrays of {0,1} (after ignoring Pulsar)
    """
    # ignore pixels where GT or Pred is Pulsar (8)
    keep = (y_true != pulsar_label) & (y_pred != pulsar_label)
    y_true_kept = y_true[keep]
    y_pred_kept = y_pred[keep]

    # binary rule: non-zero (1, 2, 6, 7) is positive
    y_true_bin = (y_true_kept != 0).astype(np.uint8)
    y_pred_bin = (y_pred_kept != 0).astype(np.uint8)
    return y_true_bin, y_pred_bin


def metrics_from_binary_cm(cm2: np.ndarray):
    """Compute Accuracy/Precision/Recall/F1 from a 2x2 confusion matrix.

    Matrix layout (sklearn default with labels=[0,1]):
        [[TN, FP],
         [FN, TP]]
    """
    tn, fp, fn, tp = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(total),
    }


def paired_bootstrap_ci_mean(
    deltas: np.ndarray,
    iters: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Paired bootstrap CI for the mean of deltas.

    Args:
        deltas: shape [B] where B is #blocks, delta_i = metricA_i - metricB_i.
        iters: number of bootstrap resamples.
        ci: confidence level.
        seed: RNG seed.

    Returns:
        mean_delta, (low, high)
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    B = deltas.shape[0]
    if B == 0:
        return 0.0, (0.0, 0.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(iters, B))
    means = deltas[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    return float(deltas.mean()), (low, high)


def evaluate_binary_flat(
    gt_dir: str,
    pred_dir: str,
    num_samples: int,
    pred_glob: str,
    verbose_every: int = 10,
    progress_prefix: str | None = None,
    progress_seconds: float = 1.0,
    collect_per_block: bool = False,
):
    """Evaluate a prediction directory in binary_flat mode with multiprocessing."""
    pred_pattern = os.path.join(pred_dir, pred_glob)
    pred_paths = sorted(glob.glob(pred_pattern))[:num_samples]
    print(f"Found {len(pred_paths)} prediction files in {pred_dir}. Using {mp.cpu_count()} cores.")

    total_cm2 = np.zeros((2, 2), dtype=np.int64)
    per_block_metrics = []
    
    stats = {"processed": 0, "missing_gt": 0, "size_mismatch": 0, "load_error": 0, "no_index": 0}
    is_aoflagger = "AOFlagger" in (progress_prefix or "")
    worker_func = partial(process_single_pair, gt_dir=gt_dir, is_aoflagger=is_aoflagger, eval_mode='binary_flat')

    start_t = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result, status in pool.imap_unordered(worker_func, pred_paths):
            if status == "success" and result is not None:
                idx, cm, m_i = result
                total_cm2 += cm
                if collect_per_block:
                    per_block_metrics.append((idx, m_i["accuracy"], m_i["precision"], m_i["recall"], m_i["f1"], m_i["total"]))
                stats["processed"] += 1
            elif "size_mismatch" in status:
                stats["size_mismatch"] += 1
            elif status in stats:
                stats[status] += 1
            
            if stats["processed"] > 0 and stats["processed"] % verbose_every == 0:
                elapsed = time.time() - start_t
                rate = stats["processed"] / max(elapsed, 1e-9)
                prefix = f"[{progress_prefix}] " if progress_prefix else ""
                print(f"\r{prefix}evaluated {stats['processed']}/{len(pred_paths)} | {rate:.2f} img/s", end="", flush=True)

    print(f"\nTotal evaluated: {stats['processed']}")
    if any(v > 0 for k, v in stats.items() if k != "processed"):
        print(f"Skipped detail: { {k:v for k,v in stats.items() if k!='processed'} }")

    if stats["processed"] == 0:
        return None

    metrics = metrics_from_binary_cm(total_cm2)
    out = {"cm2": total_cm2, "metrics": metrics}
    if collect_per_block:
        per_block_metrics.sort(key=lambda x: x[0])
        out["per_block"] = per_block_metrics
    return out

def evaluate_multiclass(
    gt_dir: str,
    pred_dir: str,
    num_samples: int,
    pred_glob: str,
    verbose_every: int = 10,
    progress_prefix: str | None = None,
    progress_seconds: float = 1.0,
):
    """Evaluate a prediction directory in multiclass mode with multiprocessing."""
    pred_pattern = os.path.join(pred_dir, pred_glob)
    pred_paths = sorted(glob.glob(pred_pattern))[:num_samples]
    print(f"Found {len(pred_paths)} prediction files. Using {mp.cpu_count()} cores.")

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    stats = {"processed": 0, "missing_gt": 0, "size_mismatch": 0, "load_error": 0, "no_index": 0}
    
    is_aoflagger = "AOFlagger" in (progress_prefix or "")
    worker_func = partial(process_single_pair, gt_dir=gt_dir, is_aoflagger=is_aoflagger, eval_mode='multiclass')

    start_t = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result, status in pool.imap_unordered(worker_func, pred_paths):
            if status == "success" and result is not None:
                total_cm += result # type: ignore
                stats["processed"] += 1
            elif "size_mismatch" in status:
                stats["size_mismatch"] += 1
            elif status in stats:
                stats[status] += 1
            
            if stats["processed"] > 0 and stats["processed"] % verbose_every == 0:
                elapsed = time.time() - start_t
                rate = stats["processed"] / max(elapsed, 1e-9)
                prefix = f"[{progress_prefix}] " if progress_prefix else ""
                print(f"\r{prefix}evaluated {stats['processed']}/{len(pred_paths)} | {rate:.2f} img/s", end="", flush=True)

    print(f"\nTotal evaluated: {stats['processed']}")
    if stats["processed"] == 0:
        return None

    # Compute per-class metrics
    per_class_metrics = {}
    for idx, label_val in enumerate(LABELS_LIST):
        class_name = CLASS_LABELS[label_val]

        tp = total_cm[idx, idx]
        fp = total_cm[:, idx].sum() - tp
        fn = total_cm[idx, :].sum() - tp

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_name] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }

    return {'cm': total_cm, 'per_class': per_class_metrics}

def plot_multiclass_confusion_matrix(cm, model_name, output_path):
    """Plot and save a normalized multiclass confusion matrix heatmap."""
    # 保存原始的 numpy array 到同名 .npy 文件，方便后续快速重绘
    base_png = output_path if output_path.endswith('.png') else output_path + '.png'
    base_pdf = base_png.replace('.png', '.pdf')
    npy_path = base_png.replace('.png', '.npy')
    np.save(npy_path, cm)
    print(f"Saved raw matrix data to {npy_path}")

    display_labels = ['Background', 'Horizontal', 'Vertical', 'Point', 'Block', 'Pulsar']
    # If the CM has 6 columns/rows, use them all
    cm_plot = cm[:6, :6].astype('float')
    
    row_sums = cm_plot.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = cm_plot / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=display_labels, yticklabels=display_labels, cbar_kws={'fraction':0.046, 'pad':0.04})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Multiclass Confusion Matrix (Normalized by True)\nModel: {model_name}')
    plt.tight_layout()
    try:
        plt.savefig(base_png, dpi=150)
        plt.savefig(base_pdf)
        print(f"Saved {base_png} and {base_pdf}")
    except Exception as e:
        print(f"Failed to save multiclass plots: {e}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RFI detection accuracy efficiently.")
    parser.add_argument('--do_binary', action='store_true', help='Run binary_flat evaluation')
    parser.add_argument('--do_multiclass', action='store_true', help='Run multiclass evaluation')
    parser.add_argument('--do_pulsar', action='store_true', help='Run pulsar preservation evaluation')

    parser.add_argument('--gt_dir', type=str, default=None, help='(deprecated) Directory for GT masks - now inferred from base_dir')
    # positional base directory containing mask_* subdirectories (e.g. mask_GroundTruth, mask_SegFormer, ...)
    parser.add_argument('base_dir', nargs='?', default='.', help='Base directory containing mask_* subdirectories')

    parser.add_argument('--num_samples', type=int, default=100, help='Max samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to write evaluation outputs (confusion matrices, npy files)')
    parser.add_argument('--verbose_every', type=int, default=10, help='Print progress every N evaluated files (0 to disable)')
    # (bootstrap/compare removed) -- plotting and comparisons are handled inside each evaluation branch
    args = parser.parse_args()
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # If no flags provided, exit with help guidance
    if not (args.do_binary or args.do_multiclass or args.do_pulsar):
        print("No action flags provided. Use --do_binary, --do_multiclass and/or --do_pulsar to run evaluations.")
        return

    # Resolve base directory and GT path
    base = os.path.abspath(args.base_dir)
    gt_dir = os.path.join(base, 'mask_GroundTruth')
    if not os.path.isdir(base):
        print(f"Base directory not found: {base}")
        return
    if not os.path.isdir(gt_dir):
        print(f"Ground truth directory not found under base: expected {gt_dir}")
        return

    # Note: the previous 'compare_simulation_and_compare' mode (grouped compare + bootstrap CI)
    # has been removed. Per-method comparisons and plotting are performed inside the
    # individual evaluation branches (binary_flat, multiclass, pulsar_preservation).

    # Fixed set of methods (two AI models + AOFlagger); ground truth is separate
    methods = {
        'SegFormer': os.path.join(base, 'mask_SegFormer'),
        'MiTUNet': os.path.join(base, 'mask_MiTUNet'),
        'AOFlagger': os.path.join(base, 'mask_AOFlagger'),
    }
    if args.do_pulsar:
        # Use base to find required method directories
        aoflagger_dir = os.path.join(base, 'mask_AOFlagger')
        mitunet_dir = os.path.join(base, 'mask_MiTUNet')
        segformer_dir = os.path.join(base, 'mask_SegFormer')

        # Validate directories
        for d in [gt_dir, aoflagger_dir, mitunet_dir, segformer_dir]:
            if not os.path.exists(d):
                print(f"Required directory not found: {d}")
                return

        # Pulsar preservation calculation
        GT_PULSAR_VAL = 8
        MODEL_PULSAR_IDX = 5

        def _block_id(name: str) -> Optional[int]:
            m = re.search(r"block(\d+)", name)
            return int(m.group(1)) if m else None

        def _build_index(dir_path: str) -> dict:
            idx = {}
            for fn in os.listdir(dir_path):
                if not fn.lower().endswith('.png'):
                    continue
                bid = _block_id(fn)
                if bid is None:
                    continue
                idx.setdefault(bid, fn)
            return idx

        gt_index = _build_index(gt_dir)
        ao_index = _build_index(aoflagger_dir)
        mi_index = _build_index(mitunet_dir)
        sf_index = _build_index(segformer_dir)

        common_ids = set(gt_index) & set(ao_index) & set(mi_index) & set(sf_index)
        print(f"Found files: GT={len(gt_index)}, AOFlagger={len(ao_index)}, MiTUNet={len(mi_index)}, SegFormer={len(sf_index)}")
        print(f"Common blocks across all methods: {len(common_ids)}")

        block_ids = sorted(common_ids)

        stats = {
            'SegFormer': {'tp': 0.0, 'total': 0.0},
            'MiTUNet': {'tp': 0.0, 'total': 0.0},
            'AOFlagger': {'tp': 0.0, 'total': 0.0}
        }

        missing = {"GT": 0, "AOFlagger": 0, "MiTUNet": 0, "SegFormer": 0}

        for bid in tqdm(block_ids, desc='Evaluating Pulsar Preservation'):
            gt_path = os.path.join(gt_dir, gt_index[bid])
            try:
                gt = np.array(Image.open(gt_path))
            except Exception:
                missing['GT'] += 1
                continue

            pulsar_mask = (gt == GT_PULSAR_VAL)
            pulsar_count = int(pulsar_mask.sum())

            # SegFormer
            sf_path = os.path.join(segformer_dir, sf_index[bid])
            try:
                sf_mask = np.array(Image.open(sf_path))
            except Exception:
                missing['SegFormer'] += 1
                sf_mask = None
            if sf_mask is not None:
                sf_preserved = (sf_mask == 0) | (sf_mask == MODEL_PULSAR_IDX)
                stats['SegFormer']['tp'] += int(np.sum(pulsar_mask & sf_preserved))
                stats['SegFormer']['total'] += pulsar_count

            # MiTUNet
            mi_path = os.path.join(mitunet_dir, mi_index[bid])
            try:
                mi_mask = np.array(Image.open(mi_path))
            except Exception:
                missing['MiTUNet'] += 1
                mi_mask = None
            if mi_mask is not None:
                mi_preserved = (mi_mask == 0) | (mi_mask == MODEL_PULSAR_IDX)
                stats['MiTUNet']['tp'] += int(np.sum(pulsar_mask & mi_preserved))
                stats['MiTUNet']['total'] += pulsar_count

            # AOFlagger
            ao_path = os.path.join(aoflagger_dir, ao_index[bid])
            try:
                ao_mask = np.array(Image.open(ao_path))
            except Exception:
                missing['AOFlagger'] += 1
                ao_mask = None
            if ao_mask is not None:
                ao_preserved = (ao_mask == 0)
                stats['AOFlagger']['tp'] += int(np.sum(pulsar_mask & ao_preserved))
                stats['AOFlagger']['total'] += pulsar_count

        # Prepare results and save CSV + small bar plot (png/pdf)
        rows = []
        for model_name, data in stats.items():
            total = int(data['total'])
            tp = int(data['tp'])
            recall = (data['tp'] / data['total']) if data['total'] > 0 else 0.0
            rows.append({'Method': model_name, 'Pulsar Pixels Total': total, 'Pulsar Pixels Preserved': tp, 'Preservation_Recall': recall})

        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, 'pulsar_preservation.csv')
        df.to_csv(csv_path, index=False)
        print(f"Wrote pulsar preservation CSV to {csv_path}")

        # Print table to console
        print('\n' + '='*60)
        print('         PULSAR PRESERVATION QUANTITATIVE ANALYSIS')
        print('='*60)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print('No pulsar pixels found in the common set.')
        print('='*60)
        print("Note: For AI models, 'Preserved' means pixels classified as 'bkg' or 'pulsar'.")
        print("      For AOFlagger, 'Preserved' means pixels NOT flagged as RFI.")
        print(f"Missing/Unreadable files during loop: {missing}")

        # Bar plot of preservation percentage
        try:
            plt.figure(figsize=(6, 4))
            method_names = df['Method'].tolist()
            vals = (df['Preservation_Recall'].astype(float) * 100).tolist()
            sns.barplot(x=method_names, y=vals)
            plt.ylabel('Preservation %')
            plt.ylim(0, 100)
            plt.title('Pulsar Preservation (%)')
            plt.tight_layout()
            png_path = os.path.join(args.output_dir, 'pulsar_preservation.png')
            pdf_path = os.path.join(args.output_dir, 'pulsar_preservation.pdf')
            plt.savefig(png_path, dpi=150)
            plt.savefig(pdf_path)
            plt.close()
            print(f"Saved plots: {png_path}, {pdf_path}")
        except Exception as e:
            print(f"Failed to save pulsar preservation plot: {e}")
        # end pulsar branch
        # continue to next actions

    if args.do_binary:
        for name, pred_dir in methods.items():
            if not os.path.isdir(pred_dir):
                print(f"Skipping {name}: directory not found: {pred_dir}")
                continue
            pred_glob = detect_pred_glob(pred_dir)
            print(f"Evaluating binary for {name} using glob '{pred_glob}'")
            res = evaluate_binary_flat(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                num_samples=args.num_samples,
                pred_glob=pred_glob,
                verbose_every=args.verbose_every,
                progress_prefix=name,
                progress_seconds=1.0,
            )
            if not res:
                print(f"No results for {name}")
                continue
            m = res['metrics']
            cm2 = res['cm2']

            # Save metrics CSV and cm npy (use method name as prefix)
            prefix = name
            metrics_df = pd.DataFrame([m])
            metrics_csv = os.path.join(args.output_dir, f"{prefix}_binary_metrics.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"Wrote binary metrics CSV to {metrics_csv}")

            cm2_npy = os.path.join(args.output_dir, f"{prefix}_binary_cm2.npy")
            np.save(cm2_npy, cm2)
            print(f"Saved binary confusion matrix (npy) to {cm2_npy}")

            # Plot binary confusion matrix (PNG + PDF)
            cm2f = cm2.astype(float)
            row_sums = cm2f.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm2f / row_sums
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=['Neg(0)', 'Pos(!=0)'],
                yticklabels=['Neg(0)', 'Pos(!=0)'],
                cbar_kws={'fraction':0.046, 'pad':0.04}
            )
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.title(f'Binary-flat Normalized Confusion Matrix (ignore Pulsar) - {name}')
            plt.tight_layout()
            png_path = os.path.join(args.output_dir, f"{prefix}_binary_confusion.png")
            pdf_path = os.path.join(args.output_dir, f"{prefix}_binary_confusion.pdf")
            plt.savefig(png_path, dpi=150)
            plt.savefig(pdf_path)
            plt.close()
            print(f"Saved binary confusion plots for {name}: {png_path}, {pdf_path}")

    # --- multiclass legacy mode ---
    if args.do_multiclass:
        for name, pred_dir in methods.items():
            if not os.path.isdir(pred_dir):
                print(f"Skipping {name}: directory not found: {pred_dir}")
                continue
            pred_glob = detect_pred_glob(pred_dir)
            print(f"Evaluating multiclass for {name} using glob '{pred_glob}'")
            res = evaluate_multiclass(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                num_samples=args.num_samples,
                pred_glob=pred_glob,
                verbose_every=args.verbose_every,
                progress_prefix=f"{name} (multiclass)",
                progress_seconds=1.0,
            )
            if not res:
                print(f"No multiclass results for {name}")
                continue

            per_class = res['per_class']
            # Save per-class CSV
            rows = []
            for cls, vals in per_class.items():
                rows.append({'Class': cls, 'IoU': vals['iou'], 'Precision': vals['precision'], 'Recall': vals['recall'], 'F1': vals['f1'], 'tp': vals['tp'], 'fp': vals['fp'], 'fn': vals['fn']})
            per_df = pd.DataFrame(rows)
            csv_path = os.path.join(args.output_dir, f"{name}_multiclass_per_class.csv")
            per_df.to_csv(csv_path, index=False)
            print(f"Saved multiclass per-class metrics CSV to {csv_path}")

            # Save confusion matrix npy + PNG/PDF
            cm = res['cm']
            cm_npy = os.path.join(args.output_dir, f"{name}_multiclass_cm.npy")
            np.save(cm_npy, cm)
            print(f"Saved multiclass confusion matrix (npy) to {cm_npy}")
            png_base = os.path.join(args.output_dir, f"{name}_multiclass_confusion.png")
            plot_multiclass_confusion_matrix(cm, name, png_base)

if __name__ == "__main__":
    main()
