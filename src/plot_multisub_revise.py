import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import cv2
from PIL import Image
from matplotlib.patches import Rectangle
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------
# 配置类和颜色映射 (参考 src/visualize_fits.py & AI_RFI.py)
# ---------------------------------------------------------
CLASS_NAMES = {
    0: 'Background',
    1: 'Horizontal RFI',
    2: 'Vertical RFI',
    3: 'Point RFI',
    4: 'Block RFI',
    5: 'Pulsar',
    6: 'Point (Legacy)',
    7: 'Block (Legacy)',
    8: 'Pulsar (Legacy)',
}

# 转换颜色为 0-1 范围的 RGB
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),       # black
    1: (0.0, 1.0, 1.0),       # cyan
    2: (1.0, 0.0, 1.0),       # magenta
    3: (0.0, 0.5, 1.0),       # blue (matched to plot_confusion_looks.py)
    4: (1.0, 1.0, 0.0),       # yellow
    5: (0.0, 1.0, 0.0),       # green (pulsar)
    6: (0.0, 0.0, 1.0),
    7: (1.0, 0.6, 0.0),
    8: (0.0, 0.8, 0.0),
}

def normalize_data(image, k=5.0):
    """参考 AI_RFI.py 中的归一化逻辑"""
    img = image.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std <= 1e-6:
        return np.zeros_like(img)
    lo, hi = mean - k * std, mean + k * std
    img = np.clip(img, lo, hi)
    img -= lo
    img /= (hi - lo)
    return img

def load_psrfits_block(path, row_idx):
    """读取 PSRFITS 的指定行数据并转换为 (Freq, Time) 用于绘图"""
    with fitsio.FITS(path, 'r') as fits:
        hdu = fits['SUBINT'] if 'SUBINT' in fits else fits[1]
        header = hdu.read_header()
        nchan = int(header["NCHAN"])
        nsblk = int(header["NSBLK"])
        record = hdu.read(rows=[row_idx])[0]
    
    raw_data = np.asarray(record["DATA"]).astype(np.float32)
    # 尝试 reshape 为 (NSBLK, NCHAN) 即 (Time, Freq)
    try:
        arr = raw_data.reshape(nsblk, nchan)
    except ValueError:
        arr = raw_data.reshape(nchan, nsblk).T
        
    # Scale and Offset
    dat_scl = np.asarray(record["DAT_SCL"], dtype=np.float32)[:nchan]
    dat_offs = np.asarray(record["DAT_OFFS"], dtype=np.float32)[:nchan]
    arr = arr * dat_scl[np.newaxis, :] + dat_offs[np.newaxis, :]
    
    # 转置并上下翻转以符合习惯：(Freq, Time)，低频在下
    return np.flipud(arr.T)

def mask_to_rgb(mask_path, is_binary=False):
    """将 Mask 文件转换为 RGB 彩色渲染"""
    if not os.path.exists(mask_path):
        return None
    
    # 读取 Mask 图像 (通常是 uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    if is_binary:
        # 对于 AOFlagger，非 0 即为 RFI (白色显示)
        rgb[mask > 0] = (1.0, 1.0, 1.0)
    else:
        # 对于多分类模型和 GT
        # 注意：如果 mask 值为 255 代表 RFI（二值化产物），需要处理
        # 但如果是多分类 index (0-5)，则直接映射
        # 兼容旧标签（legacy）: 某些工具/模拟器会使用 6/7/8 表示 point/block/pulsar
        # 将它们映射到新的标签 3/4/5 以保持颜色一致
        legacy_map = {6: 3, 7: 4, 8: 5}
        # 只在出现这些标签时才做替换，避免影响二值化(0/255)情形
        if np.any(np.isin(mask, list(legacy_map.keys()))):
            for old, new in legacy_map.items():
                mask[mask == old] = new

        for cid, color in CLASS_COLORS.items():
            if cid == 0: continue
            rgb[mask == cid] = color
            
        # 兼容性：如果 mask 是二值化的 (0, 255)，且不是 is_binary，则全转为红色辅助显示
        if mask.max() == 255 and not np.any((mask > 0) & (mask < 255)):
             rgb[mask == 255] = (1.0, 0.0, 0.0)

    return rgb

def main():
    # 定义样本索引变量 (用于第一行模拟数据)
    block_idx = 3  # 控制读取哪个样本 (0, 1, 2, ...)

    # 根路径定义
    PAPER_EXP_DIR = "PaperExperiments"

    # 路径定义
    SIM_PATH = os.path.join(PAPER_EXP_DIR, "simulation_v4/simulation_psrfits.fits")
    # REAL_PATH 指向用户指定的 G30.00+6.44 真实数据块
    REAL_PATH = os.path.join(PAPER_EXP_DIR, f"infer_and_compare_v2/images/G30.00+6.44_20240120_block{block_idx}.fits")  
    
    # Row 1 (Sim) 的 Mask 路径
    sim_masks = {
        "AOFlagger": os.path.join(PAPER_EXP_DIR, f"simulation_v4/mask_AOFlagger/simulation_psrfits_block{block_idx:04d}.png"),
        "Model A": os.path.join(PAPER_EXP_DIR, f"simulation_v4/mask_MiTUNet/simulation_psrfits_block{block_idx:04d}.png"),
        "Model B": os.path.join(PAPER_EXP_DIR, f"simulation_v4/mask_SegFormer/simulation_psrfits_block{block_idx:04d}.png"),
        "GT": os.path.join(PAPER_EXP_DIR, f"simulation_v4/mask_GroundTruth/simulation_psrfits_block{block_idx:04d}.png")
    }
    
    # Row 2 (Real) 的 Mask 路径
    # G30.00+6.44_20240120_snapshot-M09-P4-c2048b1_block0000.png
    # 或者是 G30.00+6.44_20240120_block0.png (ReadFASTData)
    
    # 构建路径前缀
    # 注意：AOFlagger, MiTUNet, SegFormer 使用了完整原始文件名+blockXXXX
    # ReadFASTData 使用了简化文件名+blockX
    
    # 原始完整文件名推测 (根据 ls 输出)
    raw_basename_full = "G30.00+6.44_20240120_snapshot-M09-P4-c2048b1"
    # ReadFASTData 文件名推测
    readfast_basename = "G30.00+6.44_20240120"
    
    real_block_suffix = f"_block{block_idx:04d}.png"  # e.g. _block0000.png
    readfast_suffix = f"_block{block_idx}.png"        # e.g. _block0.png
    
    real_masks = {
        "AOFlagger": os.path.join(PAPER_EXP_DIR, f"infer_and_compare_v2/mask_AOFlagger/{raw_basename_full}{real_block_suffix}"),
        "MiTUNet": os.path.join(PAPER_EXP_DIR, f"infer_and_compare_v2/mask_MiTUNet/{raw_basename_full}{real_block_suffix}"),
        "SegFormer": os.path.join(PAPER_EXP_DIR, f"infer_and_compare_v2/mask_SegFormer/{raw_basename_full}{real_block_suffix}"),
        "ReadFASTData": os.path.join(PAPER_EXP_DIR, f"infer_and_compare_v2/mask_ReadFASTData/{readfast_basename}{readfast_suffix}")
    }

    # 读取数据
    print(f"Loading Row 1: {SIM_PATH}...")
    try:
        img1 = load_psrfits_block(SIM_PATH, block_idx)
        img1 += 25.0  # 为 Simulated Data 加上 25 的偏移量
    except Exception as e:
        print(f"Error loading SIM_PATH: {e}")
        img1 = np.zeros((100, 100)) # dummy
        
    # 计算显示范围 (vmin, vmax)
    def get_range(data, k=3.0):
        return data.mean() - k * data.std(), data.mean() + k * data.std()

    vmin1, vmax1 = get_range(img1, k=3.0)

    print(f"Loading Row 2: {REAL_PATH}...")
    # 注意：output/G30.00+6.44_20240120_block0.fits 只有一个SUBINT行，所以row_idx=0
    try:
        img2 = load_psrfits_block(REAL_PATH, 0)  
    except Exception as e:
         print(f"Error loading REAL_PATH: {e}")
         img2 = np.zeros((100, 100)) # dummy

    vmin2, vmax2 = get_range(img2, k=3.0)

    # 绘图
    fig, axes = plt.subplots(2, 5, figsize=(24, 12))
    # 减小 wspace 以便让子图有更多横向扩展空间
    plt.subplots_adjust(top=0.9, wspace=0.35, hspace=0.3)
    
    # 手动调整子图位置和宽度：增加宽度并处理第一列 colorbar 的间距
    for i in range(5):
        for j in range(2):
            ax = axes[j, i]
            pos = ax.get_position()
            # 将每个子图宽度增加约 10%
            new_width = pos.width * 1.1
            if i > 0:
                # 第2-5列：向右偏移 0.03 以避开第一列的 colorbar，同时应用新宽度
                ax.set_position([pos.x0 + 0.03, pos.y0, new_width, pos.height])
            else:
                # 第1列：直接应用新宽度
                ax.set_position([pos.x0, pos.y0, new_width, pos.height])
    
    # 坐标范围定义
    extent = [0, 0.0533, 1031.25, 1468.75]
    
    # 更新标题 (针对第二行逻辑)
    # 第一行标题保持不变 (对应 sim_masks keys)
    titles_row1 = ["Sample", "AOFlagger", "Model A: MiT-B2+U-Net", "Model B: SegFormer-B2", "Ground Truth"]
    # 第二行标题 (对应 real_masks keys)
    titles_row2 = ["Sample", "AOFlagger", "Model A: MiT-B2+U-Net", "Model B: SegFormer-B2", "Training Annotation"]

    # 设置每行子图的轴联动
    for i in range(1, 5):
        axes[0, i].sharex(axes[0, 0])
        axes[0, i].sharey(axes[0, 0])
        axes[1, i].sharex(axes[1, 0])
        axes[1, i].sharey(axes[1, 0])

    # --- 第一行: 模拟数据 ---
    im1 = axes[0, 0].imshow(img1, aspect='auto', cmap='gist_heat', extent=extent, vmin=vmin1, vmax=vmax1)
    axes[0, 0].set_title("Simulated Data", fontweight='bold', fontsize=16)
    
    # Add colorbar for simulated data
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label('Intensity', fontsize=18)  # 调整为与其他标签一致的字号
    
    for i, (name, path) in enumerate(sim_masks.items()):
        is_binary = (name == "AOFlagger")
        rgb = mask_to_rgb(path, is_binary=is_binary)
        if rgb is not None:
            axes[0, i+1].imshow(rgb, aspect='auto', extent=extent)
        else:
            axes[0, i+1].text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}", 
                             ha='center', va='center', transform=axes[0, i+1].transAxes)
        axes[0, i+1].set_title(titles_row1[i+1], fontsize=16)

    # --- 第二行: 真实数据 ---
    im2 = axes[1, 0].imshow(img2, aspect='auto', cmap='gist_heat', extent=extent, vmin=vmin2, vmax=vmax2)
    axes[1, 0].set_title("Real Data", fontweight='bold', fontsize=16)
    
    # Add colorbar for real data
    divider2 = make_axes_locatable(axes[1, 0])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label('Intensity', fontsize=18)  # 调整为与其他标签一致的字号
    
    for i, (name, path) in enumerate(real_masks.items()):
        # AOFlagger 显示为红色 (二值化)
        # MiTUNet, SegFormer, ReadFASTData 显示彩色 (多分类)
        is_binary = (name == "AOFlagger")
        
        rgb = mask_to_rgb(path, is_binary=is_binary)
        if rgb is not None:
            axes[1, i+1].imshow(rgb, aspect='auto', extent=extent)
        else:
            axes[1, i+1].text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}", 
                             ha='center', va='center', transform=axes[1, i+1].transAxes)
        axes[1, i+1].set_title(titles_row2[i+1], fontsize=16)

    # 设置坐标轴及其标签
    import matplotlib.ticker as ticker
    for r in range(2):
        for c in range(5):
            ax = axes[r, c]
            # 仅在第二行显示 Time (s)
            if r == 1:
                ax.set_xlabel("Time (s)", fontsize=18)
            else:
                ax.set_xlabel("")
            
            # 仅在第一列显示 Frequency (MHz)
            if c == 0:
                ax.set_ylabel("Frequency (MHz)", fontsize=18)
            else:
                ax.set_ylabel("")
            
            ax.tick_params(axis='both', which='major', labelsize=16)

    # 调整加粗子图标题字体大小
    for i, ax in enumerate(axes.flatten()):
        if i < 5:  # 第一行标题
            ax.set_title(titles_row1[i], fontsize=18, fontweight='bold')  # 字体稍微变小
        elif i >= 5:  # 第二行标题
            ax.set_title(titles_row2[i - 5], fontsize=18, fontweight='bold')  # 字体稍微变小

    # 修改图例，移除 "Classes:" 前的空白区域框，并为色块添加黑色边框
    handles = [Rectangle((0,0),1,1, facecolor=CLASS_COLORS[cid], edgecolor='black', linewidth=1) for cid in range(1,6)]
    labels = [CLASS_NAMES[cid] for cid in range(1,6)]

    # 上移图例并与第一行隔出一定空间
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=len(labels), fontsize=16, frameon=True, handletextpad=0.5, columnspacing=1.0)

    # 调整整个图表纵向压缩到 80%
    fig.set_figheight(fig.get_figheight() * 0.8)
    
    save_path = "results/model_comparison_plot.pdf"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight') 
    print(f"Plot saved to: {save_path}")
    # plt.show() # 取消弹窗展示

if __name__ == "__main__":
    main()
