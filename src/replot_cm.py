#!/usr/bin/env python3
"""
快速重绘混淆矩阵的独立脚本。
用法: python src/replot_cm.py
说明: 无需重新评估数据，直接读取上次评估保存的 .npy 文件进行绘图样式调整。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_multiclass_from_npy(npy_path, model_name, output_path):
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    # 读取保存的混淆矩阵
    cm = np.load(npy_path)
    
    # 【1. 在这里修改类别标签】（如果需要剔除Pulsar只要切片 cm[:5, :5] 并删除 'Pulsar' 标签即可）
    display_labels = ['Bkg', 'Horizontal', 'Vertical', 'Point', 'Block', 'Pulsar']
    cm_plot = cm[:6, :6].astype('float')
    
    # 归一化处理
    row_sums = cm_plot.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = cm_plot / row_sums

    # 【2. 在这里调整画布和热力图样式】
    plt.figure(figsize=(10, 8)) # 调整图片尺寸
    sns.set_theme(style="white") # 移除 font_scale，通过后续显式指定字体大小

    # 根据模型名称设定自定义标题
    if model_name == 'MiTUNet':
        title_str = "Confusion Matrix for Model A (MiT-B2 + U-Net)"
    elif model_name == 'SegFormer':
        title_str = "Confusion Matrix for Model B (SegFormer-B2)"
    else:
        title_str = f"Confusion Matrix for {model_name}"

    # cbar_kws 设置可以调整右侧 colorbar
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=0.5, linecolor='lightgray') # 加大色块数字字号并加粗，添加内部浅色分割线
    
    # 获取colorbar修改其边框和刻度字号
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)
        cbar.outline.set_edgecolor('black')
        # 增大 colorbar 的刻度字号 (0.00, 0.20, ... 1.00)
        cbar.ax.tick_params(labelsize=14) 

    # 给矩阵外边框加上一圈黑线
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 【3. 在这里调整标题和轴标签】
    plt.xticks(fontsize=14, fontweight='bold') # 恢复 fontWeight='bold'
    plt.yticks(fontsize=14, fontweight='bold') # 恢复 fontWeight='bold'
    plt.ylabel('True Label', fontweight='bold', fontsize=18)
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=18)
    # 将标题字号从 16 增大到 20 或更大
    plt.title(title_str, pad=20, fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    # dpi参数对pdf影响不大，但如果是栅格化的元素会起作用。可直接保存为pdf
    plt.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight') 
    print(f"Saved replotted matrix to {output_path}")

def main():
    # 数据目录 - 如果你换了其他目录，在这里修改
    base_dir = "/home/cbm/deRFI/simulation_v2"
    
    models = ['SegFormer', 'MiTUNet']
    
    for model in models:
        npy_path = os.path.join(base_dir, f"confusion_matrix_multiclass_{model}.npy")
        
        # 修改输出的文件名及格式为pdf
        if model == 'MiTUNet':
            out_name = "ConfusionMatrix_Model_A.pdf"
        elif model == 'SegFormer':
            out_name = "ConfusionMatrix_Model_B.pdf"
        else:
            out_name = f"ConfusionMatrix_{model}.pdf"
            
        out_path = os.path.join(base_dir, out_name)
        
        print(f"Processing {model} from {npy_path}")
        plot_multiclass_from_npy(npy_path, model, out_path)

if __name__ == "__main__":
    main()
