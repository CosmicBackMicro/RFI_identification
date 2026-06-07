#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_custom_cm(cm_data, display_labels, title_str, output_path):
    cm_plot = np.array(cm_data).astype('float')
    
    # 归一化处理 (按行归一化)
    row_sums = cm_plot.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = cm_plot / row_sums

    # 绘图样式
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    ax = sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=0.5, linecolor='lightgray')
    
    # 获取colorbar并修改样式
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)
        cbar.outline.set_edgecolor('black')
        cbar.ax.tick_params(labelsize=12) 

    # 矩阵外边框黑线
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 标题和轴标签
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold', fontsize=16)
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=16)
    plt.title(title_str, pad=20, fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    print(f"Saved custom matrix to {output_path}")

def main():
    # 用户提供的数据
    # cm_data = [
    #     [0.963, 0.028, 0.000, 0.001, 0.001, 0.007],   # bkg
    #     [0.059, 0.934, 0.000, 0.001, 0.001, 0.006],   # horizontal
    #     [0.038, 0.002, 0.944, 0.001, 0.002, 0.013],   # vertical
    #     [0.190, 0.039, 0.001, 0.760, 0.005, 0.005],   # point
    #     [0.071, 0.021, 0.000, 0.006, 0.892, 0.010],   # block
    #     [0.040, 0.011, 0.000, 0.000, 0.000, 0.948]    # pulsar
    # ]

    # cm_data = [
    #     [0.958, 0.033, 0.000, 0.001, 0.001, 0.007],   # bkg
    #     [0.060, 0.933, 0.000, 0.001, 0.001, 0.005],   # horizontal
    #     [0.031, 0.002, 0.950, 0.001, 0.001, 0.015],   # vertical
    #     [0.130, 0.047, 0.000, 0.815, 0.003, 0.005],   # point
    #     [0.033, 0.039, 0.000, 0.010, 0.908, 0.010],   # block
    #     [0.038, 0.013, 0.000, 0.000, 0.000, 0.949]    # pulsar
    # ]
    
    
    cm_data = [
        [0.966, 0.022, 0.001, 0.001, 0.001, 0.010],   # bkg
        [0.154, 0.832, 0.000, 0.002, 0.001, 0.011],   # horizontal
        [0.071, 0.003, 0.898, 0.001, 0.001, 0.027],   # vertical
        [0.541, 0.056, 0.002, 0.384, 0.005, 0.011],   # point
        [0.160, 0.116, 0.000, 0.028, 0.661, 0.034],   # block
        [0.060, 0.007, 0.001, 0.001, 0.001, 0.931]    # pulsar
    ]

    labels = ['Bkg', 'Horizontal', 'Vertical', 'Point', 'Block', 'Pulsar']
    # title = "Confusion Matrix for Model A (MiT-B2 + U-Net)"
    title = "Confusion Matrix for Model B (SegFormer-B2)"
    output = "/home/cbm/deRFI/results/ConfusionMatrix_Model_B.pdf"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    plot_custom_cm(cm_data, labels, title, output)

if __name__ == "__main__":
    main()
