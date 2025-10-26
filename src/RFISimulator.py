#!/usr/bin/env python3
"""
RFISimulator: 生成用于RFI检测的全合成时频数据

- 分步骤生成：
  1. 生成高斯背景
  2. 添加带状RFI
  3. 添加点状RFI
  4. 保存图像和掩码PNG
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import fitsio
import matplotlib.gridspec as gridspec
import time
import multiprocessing
import os

BKG = 0             # 背景
CHAN_ALL = 1        # 带状RFI粗类
POINT_ALL = 2       # 点状RFI粗类
CHAN_BRIGHT = 3     # 带状亮条纹
CHAN_DARK = 4       # 带状暗条纹
CHAN_COMPLEX = 5    # 带状复杂条纹
POINT_RANDOM = 6    # 点状RFI
POINT_PERIOD = 7    # 周期性RFI
POINT_BLOCK = 8     # 块状RFI

def generate_background(nsamp: int, nchan: int, bg_mu: float, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成高斯背景和空的mask，形状 (nsamp, nchan)。"""
    rng = np.random.default_rng(seed if seed != 0 else None)
    data = rng.normal(loc=bg_mu, scale=bg_sigma, size=(nsamp, nchan)).astype(np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    return data, mask


def add_many_periodic_points(data: np.ndarray, mask: np.ndarray, count: int, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """添加多个参数随机的周期性点状RFI。"""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    
    for i in range(count):
        # 为每个周期性RFI随机化参数
        rand_amp_sigma = rng.random() * 4.0 + 1.0  # 幅度在 [1, 5) sigma
        rand_width = rng.integers(1, 20)            # 宽度在 [1, 20)
        rand_duty = rng.random() * 0.1 + 0.05      # 占空比在 [0.05, 0.15)

        amp_val = rand_amp_sigma * bg_sigma
        flag = POINT_PERIOD if rand_duty < 0.95 else CHAN_BRIGHT

        start_ch = int(rng.integers(0, max(1, nchan - rand_width + 1)))
        end_ch = min(nchan, start_ch + rand_width)
        on = rng.random(nsamp) < rand_duty
        
        print(f"Adding periodic point RFI ({i+1}/{count}): amp_sigma={rand_amp_sigma:.2f}, width={rand_width}, duty={rand_duty:.3f}, start_chan={start_ch}")

        data[on, start_ch:end_ch] += amp_val
        # 标记整行，而不是离散的点
        mask[:, start_ch:end_ch] = flag
    return data, mask

def add_chan(
        data: np.ndarray, mask: np.ndarray,
        amplitude: float, center_chan: float, std_dev: float, width: int,
        mode: str = 'gaussian'
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    添加带状RFI（高斯或均匀分布），返回 (data, mask)。
    参数：
        mode: 'gaussian' 用高斯分布，'uniform' 用均匀分布
    """
    nsamp, nchan = data.shape
    if width % 2 == 0:
        raise ValueError("Width must be odd.")
    half_width = (width - 1) // 2
    start_ch = int(max(0, np.floor(center_chan) - half_width))
    end_ch = int(min(nchan, start_ch + width))
    CHAN_SUBTYPE = CHAN_BRIGHT if amplitude > 0 else CHAN_DARK
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 使用NumPy向量化操作替代Python循环以提高性能
    if mode == 'gaussian':
        # 1. 创建作用范围内的通道索引
        ch_indices = np.arange(start_ch, end_ch)
        # 2. 计算一维高斯剖面
        gauss_profile = np.exp(-0.5 * ((ch_indices - center_chan) / std_dev) ** 2)
        # 3. 使用广播将一维剖面应用到整个二维数据区域
        data[:, start_ch:end_ch] += gauss_profile * amplitude
        
    elif mode == 'uniform':
        # 对于均匀分布，直接在切片上操作
        data[:, start_ch:end_ch] += amplitude
    else:
        raise ValueError(f"Unknown mode: {mode}, must be 'gaussian' or 'uniform'")

    # 统一更新mask
    mask[:, start_ch:end_ch] = CHAN_SUBTYPE
    
    return data, mask
        
def add_many_random_points(
        data: np.ndarray, mask: np.ndarray, point_count: int, point_amp_sigma: float, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """添加点状RFI，返回 (data, mask)。在已有 mask 上更新点状标签（使用 POINT_RANDOM）。

    Parameters:
    - data: 2D array (nsamp, nchan)
    - mask: existing mask array to update (will be modified in-place)
    - point_count: number of point RFI to add
    - point_amp_sigma, bg_sigma: amplitude scaling
    - seed: RNG seed
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    # 保证 mask 存在并为 uint8
    if mask is None:
        mask = np.zeros_like(data, dtype=np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if point_count > 0:
        # 只从未被标记的像素（mask==0）中选取
        available = np.where(mask == 0)
        total_avail = available[0].size
        if total_avail < point_count:
            print(f"Warning: only {total_avail} unmarked pixels, but point_count={point_count}. Will use all available.")
            select_count = total_avail
        else:
            select_count = point_count
        idx = rng.choice(total_avail, size=select_count, replace=False)
        ts = available[0][idx]
        cs = available[1][idx]
        data[ts, cs] += (point_amp_sigma * bg_sigma)
        mask[ts, cs] = POINT_RANDOM
    return data, mask


def add_blob_point(
        data: np.ndarray, mask: np.ndarray,
        amplitude: float, x: int, y: int, sigma_x: float, sigma_y: float,
        mask_threshold_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    为背景图像加上一个二维高斯分布的点状RFI (blob)。

    参数:
    - amplitude: 高斯分布的峰值幅度
    - x, y: 中心像素坐标 (时间, 通道)
    - sigma_x, sigma_y: x和y方向上的标准差 (像素单位)
    - mask_threshold_ratio: 用于标记mask的阈值，相对于峰值幅度
    """
    nsamp, nchan = data.shape

    # 定义高斯函数影响的窗口范围，取3倍sigma确保覆盖大部分能量
    win_x_half = int(np.ceil(3 * sigma_x))
    win_y_half = int(np.ceil(3 * sigma_y))

    start_x = max(0, x - win_x_half)
    end_x = min(nsamp, x + win_x_half + 1)
    start_y = max(0, y - win_y_half)
    end_y = min(nchan, y + win_y_half + 1)

    # 如果窗口无效则直接返回
    if start_x >= end_x or start_y >= end_y:
        return data, mask

    # 创建窗口内的坐标网格
    xx, yy = np.meshgrid(np.arange(start_y, end_y), np.arange(start_x, end_x))

    # 计算二维高斯分布
    gauss_val = amplitude * np.exp(-0.5 * (((yy - x) / sigma_x)**2 + ((xx - y) / sigma_y)**2))

    # 将高斯信号添加到数据中
    data[start_x:end_x, start_y:end_y] += gauss_val.astype(data.dtype)
    
    # 根据阈值确定要标记的区域
    threshold = amplitude * mask_threshold_ratio
    mask_window = gauss_val > threshold
    
    # 更新mask，仅标记超过阈值的像素
    mask[start_x:end_x, start_y:end_y][mask_window] = POINT_RANDOM
    
    print(f"Adding blob point RFI at ({x}, {y}) with amplitude={amplitude:.2f}, sigma=({sigma_x:.2f}, {sigma_y:.2f})")

    return data, mask


def add_many_blob_points(
        data: np.ndarray, mask: np.ndarray,
        count: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    添加多个参数随机的二维高斯分布RFI (blob)。
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape

    for i in range(count):
        # 为每个blob RFI随机化参数
        rand_x = rng.integers(0, nsamp)
        rand_y = rng.integers(0, nchan)
        rand_amplitude = rng.uniform(10, 50) * bg_sigma
        rand_sigma_x = rng.uniform(1.5, 2.0)
        rand_sigma_y = rng.uniform(1.5, 4.0)

        print(f"Preparing multi-blob RFI ({i+1}/{count}):")
        data, mask = add_blob_point(
            data, mask,
            amplitude=rand_amplitude,
            x=rand_x, y=rand_y,
            sigma_x=rand_sigma_x, sigma_y=rand_sigma_y
        )
    
    return data, mask


def plot_synth_result(data: np.ndarray, mask: np.ndarray, plot_path: str) -> None:
    """绘制成果展示用的图表，包含积分曲线，保存为PNG。"""
    fig = plt.figure(figsize=(15, 8))  # 增加宽度以容纳mask
    gs = gridspec.GridSpec(
        3, 6, 
        hspace=0, wspace=0, 
        figure=fig,
        width_ratios =  [0.5, 1, 1, 0.1, 0.2, 2],
        height_ratios = [0.7, 1, 1]
    )

    labels_fontsize = 10
    # 时间积分：上方，跨两列
    ax_time = fig.add_subplot(gs[0, 1:3])
    time_profile = data.mean(axis=1)  # (nsamp,)
    ax_time.plot(np.arange(data.shape[0]), time_profile, color='blue', linewidth=0.5)
    ax_time.set_title('Time Profile', fontsize=labels_fontsize)
    ax_time.set_xlabel('Time samples', fontsize=labels_fontsize)
    ax_time.set_ylabel('Mean', fontsize=labels_fontsize)
    ax_time.tick_params(axis='x', which='both', direction='in', bottom=True, top=True, labeltop=True)

    # 频率积分：左方，跨两行
    ax_freq = fig.add_subplot(gs[1:3, 0])
    freq_profile = data.mean(axis=0)  # (nchan,)
    ax_freq.plot(freq_profile, np.arange(data.shape[1]), color='red', linewidth=0.5)
    ax_freq.set_title('Frequency Profile', fontsize=labels_fontsize)
    ax_freq.set_ylabel('Frequency channels', fontsize=labels_fontsize)
    ax_freq.set_xlabel('Mean', fontsize=labels_fontsize)

    # 数据图像：中间，跨两行两列
    ax_data = fig.add_subplot(gs[1:3, 1:3])
    im1 = ax_data.imshow(
        data.T, 
        aspect='auto', 
        origin='lower', 
        cmap='gist_heat', 
        vmin=float(np.percentile(data, 0.27)), 
        vmax=float(np.percentile(data, 99.73))
    )
    ax_data.set_xlabel('Time samples')
    ax_data.set_yticklabels([])

    # colorbar 在右侧
    cbar_ax = fig.add_subplot(gs[1:3, 3])
    plt.colorbar(im1, cax=cbar_ax, orientation='vertical')

    # mask图像：最右侧，跨两行
    mask_cmap = 'tab10'
    ax_mask = fig.add_subplot(gs[1:3, 5])
    # 使用 Pastel1 色板以便可视化各 mask 类别
    im_mask = ax_mask.imshow(
        mask.T,
        aspect='auto',
        origin='lower',
        cmap=mask_cmap,
        vmin=0,
        vmax=8,
        interpolation='none'
    )
    ax_mask.set_title('Mask')
    ax_mask.set_xlabel('Time samples')
    ax_mask.set_ylabel('Frequency channels')
    ax_mask.set_yticklabels([])  # 隐藏y标签以节省空间

    # 添加颜色图例（legend）
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    # 类别及标签（可自定义）
    mask_labels = {
        0: 'Background',
        1: 'Chan RFI (coarse)',
        2: 'Point RFI (coarse)',
        3: 'Chan RFI - Bright',
        4: 'Chan RFI - Dark',
        5: 'Chan RFI - Complex',
        6: 'Point RFI - Random',
        7: 'Point RFI - Periodic',
        8: 'Point RFI - Block',
    }
    # 获取 colormap 和归一化规则，确保与 imshow 一致
    cmap = plt.get_cmap(mask_cmap)
    norm = mcolors.Normalize(vmin=0, vmax=8)
    
    handles = []
    for k, v in mask_labels.items():
        color = cmap(norm(k))
        patch = mpatches.Patch(color=color, label=f'{k}: {v}')
        handles.append(patch)
    ax_mask.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8, frameon=False)

    # 对齐轴范围（不共享轴，避免标签冲突）
    ax_time.set_xlim(ax_data.get_xlim())  # 时间轮廓x轴范围与主图相同
    ax_freq.set_ylim(ax_data.get_ylim())  # 频率轮廓y轴范围与主图相同
    ax_mask.set_xlim(ax_data.get_xlim())  # mask x轴与主图相同
    ax_mask.set_ylim(ax_data.get_ylim())  # mask y轴与主图相同

    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")




def save_image_mask(data: np.ndarray, mask: np.ndarray, fits_path: str, mask_png_path: str) -> None:
    """保存数据为FITS格式（匹配C函数格式），mask为PNG格式。"""
    nsamp, nchan = data.shape

    # --- 调试信息: mask 类别计数与占比 ---
    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    print("Mask summary before saving:")
    for u, c in zip(unique, counts):
        pct = (c / total_pixels) * 100.0
        print(f"  class {int(u)}: {c} pixels ({pct:.3f}%)")
    # --- end debug ---
    
    # 计算量化参数（全局min/max，每个通道相同）
    data_min = float(data.min())
    data_max = float(data.max())
    scale_val = (data_max - data_min) / 255.0 if data_max > data_min else 1.0
    offset_val = data_min
    
    scale = np.full(nchan, scale_val, dtype=np.float32)
    offset = np.full(nchan, offset_val, dtype=np.float32)
    quantized = ((data - offset_val) / scale_val).clip(0, 255).astype(np.uint8)
    data_flat = quantized.flatten()
    
    # 创建FITS文件并写入二进制表
    fits = fitsio.FITS(fits_path, mode='rw', clobber=True)
    
    # 定义表结构
    dtype = np.dtype([
        ('DAT_OFFS', 'f4', nchan),
        ('DAT_SCL', 'f4', nchan),
        ('DATA', 'u1', nsamp * nchan)
    ])
    fits.create_table_hdu(dtype=dtype, extname='SUBINT')
    
    # 创建结构化数组并写入
    row = np.array([(offset, scale, data_flat)], dtype=dtype)
    fits[1].write(row)
    
    # 写入头部关键字（匹配C函数）
    fits[1].write_key('TBIN', 4.9152e-05, 'Time per sample (s)')  # 默认值
    fits[1].write_key('CHAN_BW', 0.4882812, 'Channel bandwidth (MHz)')
    fits[1].write_key('NSBLK', nsamp, 'Samples per block')
    fits[1].write_key('NCHAN', nchan, 'Frequency channels')
    fits[1].write_key('ORIGIN', 'RFISimulator', 'Source filename')  # 默认
    fits[1].write_key('BLOCKIDX', 0, 'Original block index')  # 默认
    fits[1].write_key('NBLOCKS', 1, 'Number of blocks per read')  # 默认
    
    # 列单位
    fits[1].write_key('TUNIT1', '', 'units for DAT_OFFS')
    fits[1].write_key('TUNIT2', '', 'units for DAT_SCL')
    fits[1].write_key('TUNIT3', 'RAW', 'units for DATA')
    
    from datetime import datetime
    date_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    fits[1].write_key('DATE', date_str, 'File creation date')
    
    fits.close()
    print(f"Data saved as FITS: {fits_path}")
    
    # 保存 mask 为纯像素 PNG（每个像素即为 mask 类别）
    try:
        from PIL import Image
        # 转置回原始 (nsamp, nchan) 布局 -> PNG 期望 (width, height) 对应 (nchan, nsamp)
        mask_img = (mask.T).astype(np.uint8)
        mask_img = np.flipud(mask_img)
        # 直接保存灰度图（每个像素为类标）。如果你想使用调色板，可在这里添加 palette 支持。
        im = Image.fromarray(mask_img, mode='L')
        im.save(mask_png_path, format='PNG')
        print(f"Raw mask PNG saved (PIL): {mask_png_path}")
    except Exception:
        # 回退到 matplotlib 的 imsave（不会有轴或标题）
        import matplotlib.image as mpimg
        # 对mask进行垂直翻转，以匹配FITS图像加载后的方向
        flipped_mask = np.flipud(mask)
        mpimg.imsave(mask_png_path, flipped_mask.T, cmap='gray', vmin=0, vmax=8)
        print(f"Raw mask PNG saved (matplotlib): {mask_png_path}")


def add_many_chans(
        data: np.ndarray, mask: np.ndarray,
        n_gaussian: int, n_uniform: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多个随机参数的带状RFI。

    可以手动指定 'gaussian' 和 'uniform' 模式的RFI数量。
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    _, nchan = data.shape

    # 确定每个RFI的模式
    rfi_modes = []
    rfi_modes.extend(['gaussian'] * n_gaussian)
    rfi_modes.extend(['uniform'] * n_uniform)
    
    rng.shuffle(rfi_modes)  # 随机打乱顺序
    
    count = n_gaussian + n_uniform

    for i, mode in enumerate(rfi_modes):
        # 生成随机参数
        rand_std_dev = rng.random() * 3.0 + 2.0  # std_dev 在 [2.0, 5.0)
        # 从 [-5, -1) U [1, 5) sigma 范围内抽样，以生成亮暗两种条纹
        while True:
            rand_amplitude_in_sigma = (rng.random() * 10.0 - 5.0)  # 范围 [-5.0, 5.0)
            if abs(rand_amplitude_in_sigma) >= 1.0:
                break
        rand_amplitude = rand_amplitude_in_sigma * bg_sigma
        
        rand_center_chan = rng.integers(0, nchan)
        rand_width = rng.integers(3, 26)  # 宽度在 3 到 25 之间
        if rand_width % 2 == 0:
            rand_width += 1  # 确保宽度为奇数
        
        print(f"Adding random chan RFI ({i+1}/{count}): mode={mode}, amplitude={rand_amplitude:.2f}, center_chan={rand_center_chan}, std_dev={rand_std_dev:.2f}, width={rand_width}")
        
        data, mask = add_chan(data, mask, rand_amplitude, rand_center_chan, rand_std_dev, rand_width, mode=mode)
    
    return data, mask


def generate_single_sample(args):
    """
    生成单个合成RFI数据样本的包装函数，用于多进程。
    """
    # 从元组中解包参数
    nsamp, nchan, bg_mu, bg_sigma, seed, plot_out, fits_out, mask_png_out, do_plot = args
    
    print(f"--- Generating sample with seed {seed} ---")
    # 步骤1: 生成背景和初始mask
    data, mask = generate_background(nsamp, nchan, bg_mu, bg_sigma, seed)
    print("Background and initial mask generated.")

    # 步骤2: 添加周期性点状RFI
    data, mask = add_many_periodic_points(data, mask, count=2, bg_sigma=bg_sigma, seed=seed + 1)
    print("Periodic point RFI added.")
    
    # 步骤3: 添加参数随机的亮、暗条纹RFI
    data, mask = add_many_chans(data, mask, n_gaussian=3, n_uniform=1, bg_sigma=bg_sigma, seed=seed + 2)
    print("Multiple random chan RFIs added.")

    # 步骤4: 添加随机点状RFI
    data, mask = add_many_random_points(data, mask, point_count=1000, point_amp_sigma=6.0, bg_sigma=bg_sigma, seed=seed + 3)
    print("Random point RFI added.")

    # 步骤5: 添加多个高斯斑点RFI
    data, mask = add_many_blob_points(data, mask, count=5, bg_sigma=bg_sigma, seed=seed + 4)
    print("Multiple blob point RFIs added.")
    
    # 步骤6: (可选) 绘制成果图表
    if do_plot:
        plot_synth_result(data, mask, plot_out)
    
    # 步骤7: 保存数据和mask
    save_image_mask(data, mask, fits_out, mask_png_out)
    print(f"--- Sample with seed {seed} finished ---\n")


def main():
    # === 批量生成参数 ===
    num_samples = 6000
    base_seed = 98765 # 哼！哼！哼！啊啊啊啊啊啊
    plot_interval = 50  # 每50个样本绘制一张图
    
    # === 图片尺寸 ===
    nsamp = 4096
    nchan = 1792
    # === 背景分布 ===
    bg_mu = 0.0
    bg_sigma = 1.0
    # === 输出路径 ===
    output_dir = '/home/cbm/deRFI/output'
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # --- 多进程设置 ---
    # 准备传递给每个进程的参数列表
    tasks = []
    for i in range(num_samples):
        current_seed = base_seed + i
        
        # 为每个样本创建唯一的文件名
        plot_out = f'{output_dir}/plot_{i}.png'
        fits_out = f'{output_dir}/data_{i}.fits'
        mask_png_out = f'{output_dir}/mask_{i}.png'

        # 控制是否绘图
        do_plot = (i % plot_interval == 0)

        task_args = (
            nsamp, nchan, bg_mu, bg_sigma, current_seed,
            plot_out, fits_out, mask_png_out, do_plot
        )
        tasks.append(task_args)

    # --- 执行多进程 ---
    # 使用与CPU核心数相同的进程数，或者可以手动指定
    num_processes = multiprocessing.cpu_count()
    print(f"Starting parallel generation with {num_processes} processes...")
    
    total_start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(generate_single_sample, tasks)
        
    total_end_time = time.time()
    print(f"Finished generating {num_samples} samples in {total_end_time - total_start_time:.2f} seconds.")




if __name__ == '__main__':
    main()
