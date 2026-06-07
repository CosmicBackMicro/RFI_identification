import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import re

def calculate_pulsar_preservation():
    # 路径配置
    base_dir = "/home/cbm/deRFI/simulation_and_compare"
    gt_dir = os.path.join(base_dir, "mask_GroundTruth")
    aoflagger_dir = os.path.join(base_dir, "mask_AOFlagger")
    mitunet_dir = os.path.join(base_dir, "mask_MiTUNet")
    segformer_dir = os.path.join(base_dir, "mask_SegFormer")

    # 类别定义 (根据之前的代码映射)
    # GT 中的原始值映射: {0: bkg, 1: horizontal, 2: vertical, 6: point, 7: block, 8: pulsar}
    # 脚本内部为了匹配模型的输出索引 (0-5)，需要先定义映射
    GT_PULSAR_VAL = 8 # 原始 FITS/PNG 中的脉冲星值
    MODEL_PULSAR_IDX = 5 # 模型预测时的脉冲星索引
    
    def _block_id(name: str) -> int | None:
        """从文件名中提取 block 的数字编号。支持 block0001 / block1 两种形式。"""
        m = re.search(r"block(\d+)\.png$", name)
        return int(m.group(1)) if m else None

    def _build_index(dir_path: str) -> dict[int, str]:
        """将目录内 png 文件索引为 {block_id: filename}。"""
        idx: dict[int, str] = {}
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith(".png"):
                continue
            bid = _block_id(fn)
            if bid is None:
                continue
            # 若重复，保留字典里已有的（通常不会发生）
            idx.setdefault(bid, fn)
        return idx

    gt_index = _build_index(gt_dir)
    ao_index = _build_index(aoflagger_dir)
    mi_index = _build_index(mitunet_dir)
    sf_index = _build_index(segformer_dir)

    common_ids = set(gt_index) & set(ao_index) & set(mi_index) & set(sf_index)
    print(
        f"Found files: GT={len(gt_index)}, AOFlagger={len(ao_index)}, MiTUNet={len(mi_index)}, SegFormer={len(sf_index)}\n"
        f"Common blocks across all methods: {len(common_ids)}"
    )

    # 仅在共同集合上统计，避免因命名差异导致 total 不一致
    block_ids = sorted(common_ids)

    stats = {
        'SegFormer': {'tp': 0.0, 'total': 0.0},
        'MiTUNet': {'tp': 0.0, 'total': 0.0},
        'AOFlagger': {'tp': 0.0, 'total': 0.0}
    }

    missing = {
        "GT": 0,
        "AOFlagger": 0,
        "MiTUNet": 0,
        "SegFormer": 0,
    }

    for bid in tqdm(block_ids, desc="Evaluating Pulsar Preservation"):
        # 读取 GT
        gt_path = os.path.join(gt_dir, gt_index[bid])
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            missing["GT"] += 1
            continue
        
        # 统计脉冲星像素数量 (GT 中的原始值是 8)
        pulsar_mask = (gt == GT_PULSAR_VAL)
        pulsar_count = np.sum(pulsar_mask)
        # 即使 pulsar_count 为 0 也要继续，因为我们要验证所有模型处理的一致性（虽然 0 脉冲对 recall 没贡献）
            
        # 1. SegFormer
        sf_path = os.path.join(segformer_dir, sf_index[bid])
        sf_mask = cv2.imread(sf_path, cv2.IMREAD_UNCHANGED)
        if sf_mask is None:
            missing["SegFormer"] += 1
        else:
            sf_preserved = (sf_mask == 0) | (sf_mask == MODEL_PULSAR_IDX)
            stats['SegFormer']['tp'] += np.sum(pulsar_mask & sf_preserved)
            stats['SegFormer']['total'] += pulsar_count

        # 2. MiTUNet
        mi_path = os.path.join(mitunet_dir, mi_index[bid])
        mi_mask = cv2.imread(mi_path, cv2.IMREAD_UNCHANGED)
        if mi_mask is None:
            missing["MiTUNet"] += 1
        else:
            mi_preserved = (mi_mask == 0) | (mi_mask == MODEL_PULSAR_IDX)
            stats['MiTUNet']['tp'] += np.sum(pulsar_mask & mi_preserved)
            stats['MiTUNet']['total'] += pulsar_count

        # 3. AOFlagger
        ao_path = os.path.join(aoflagger_dir, ao_index[bid])
        ao_mask = cv2.imread(ao_path, cv2.IMREAD_UNCHANGED)
        if ao_mask is None:
            missing["AOFlagger"] += 1
        else:
            ao_preserved = (ao_mask == 0)
            stats['AOFlagger']['tp'] += np.sum(pulsar_mask & ao_preserved)
            stats['AOFlagger']['total'] += pulsar_count

    # 计算百分比
    results = []
    for model_name, data in stats.items():
        if data['total'] > 0:
            recall = data['tp'] / data['total']
            results.append({
                'Method': model_name,
                'Pulsar Pixels Total': data['total'],
                'Pulsar Pixels Preserved': data['tp'],
                'Preservation % (Recall)': f"{recall*100:.2f}%"
            })

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("         PULSAR PRESERVATION QUANTITATIVE ANALYSIS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print("Note: For AI models, 'Preserved' means pixels classified as 'bkg' or 'pulsar'.")
    print("      For AOFlagger, 'Preserved' means pixels NOT flagged as RFI.")
    print(f"Missing/Unreadable files during loop (should be 0): {missing}")

if __name__ == "__main__":
    calculate_pulsar_preservation()
