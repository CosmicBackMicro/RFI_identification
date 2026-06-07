import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(in_path, out_path):
    # 读取图像
    img = cv2.imread(in_path)
    if img is None:
        print(f"Error: Could not read {in_path}")
        return
    
    # 横向拉伸原图比例 (拉伸到 1.25 倍宽)
    stretch_factor = 1.6
    img = cv2.resize(img, (int(img.shape[1] * stretch_factor), img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 先在原图上寻找子图的边界框
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 200: 
            boxes.append((x, y, w, h))
            
    boxes = sorted(boxes, key=lambda b: (b[1]//100, b[0]))
    
    # 增加子图之间的空隙
    gap = 200  # 增加图片中间的留白
    if len(boxes) > 1:
        splits = []
        for i in range(len(boxes)-1):
            x1 = boxes[i][0] + boxes[i][2]
            x2 = boxes[i+1][0]
            splits.append((x1 + x2) // 2)
            
        # 去掉顶部 100 像素的标题区域
        crop_top = 100
        new_h = img.shape[0] - crop_top
        new_w = img.shape[1] + gap * len(splits)
        new_img = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
        
        # 底部（包含子图）按 split 切片并插入 gap
        prev_split = 0
        new_x = 0
        for i, split in enumerate(splits):
            part_w = split - prev_split
            new_img[:, new_x:new_x+part_w] = img[crop_top:, prev_split:split]
            new_x += part_w + gap
            prev_split = split
            
        new_img[:, new_x:new_x+(img.shape[1]-prev_split)] = img[crop_top:, prev_split:]
        img = new_img
        
        # 更新 boxes 的坐标 (因为去掉了顶部，y 坐标需要减去 crop_top)
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            boxes[i] = (x + i * gap, y - crop_top, w, h)
            
        # 抹掉旧的子标题 (将子图上方的区域全部涂白)
        min_y = min([b[1] for b in boxes])
        if min_y > 0:
            img[0:min_y, :] = 255
            
    # 给图像四周增加白色留白，以便显示坐标轴和刻度
    # 减小上下留白，保持左右留白足够显示标签
    pad_x, pad_y = 220, 120
    img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # 更新 boxes 的坐标 (因为加了 padding)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        boxes[i] = (x + pad_x, y + pad_y, w, h)
        
    print(f"Found {len(boxes)} panels.")
    
    # 使用 matplotlib 绘制坐标轴
    fig_w, fig_h = img.shape[1], img.shape[0]
    fig = plt.figure(figsize=(fig_w/100, fig_h/100), dpi=100)
    
    # 铺满整个画布
    ax_main = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax_main.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_main.axis('off')
    
    titles = [
        "Input Search-mode Data",
        "Multi-class Training Mask",
        "AI Identification Result",
        "Prediction Overlay"
    ]
    
    # 在每个检测到的子图区域叠加坐标轴
    for i, (x, y, w, h) in enumerate(boxes):
        # rect 是 (left, bottom, width, height) (归一化坐标)
        rect = (x/fig_w, 1 - (y+h)/fig_h, w/fig_w, h/fig_h)
        
        ax_sub = fig.add_axes(rect)
        ax_sub.set_facecolor('none') # 透明背景
        
        # 设置刻度 (0 到 1)
        # x轴刻度：0.00, 0.01, 0.02, 0.03, 0.04, 0.05
        # 映射到 0-1 范围：(val - 0) / (0.050331 - 0)
        x_min, x_max = 0.0, 0.050331
        x_ticks_val = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
        x_ticks_pos = [(v - x_min) / (x_max - x_min) for v in x_ticks_val]
        ax_sub.set_xticks(x_ticks_pos)
        
        # y轴刻度：1100, 1200, 1300, 1400
        # 映射到 0-1 范围：(val - 1031.25) / (1468.75 - 1031.25)
        y_min, y_max = 1031.25, 1468.75
        y_ticks_val = [1100, 1200, 1300, 1400]
        y_ticks_pos = [(v - y_min) / (y_max - y_min) for v in y_ticks_val]
        ax_sub.set_yticks(y_ticks_pos)
        
        # 设置刻度标签
        # x轴：0.00, 0.01, 0.02, 0.03, 0.04, 0.05
        ax_sub.set_xticklabels([f"{v:.2f}" for v in x_ticks_val])
        
        # y轴：1100, 1200, 1300, 1400
        ax_sub.set_yticklabels([str(v) for v in y_ticks_val])
        
        # 设置刻度和边框颜色，使其在深色/浅色背景上都可见
        ax_sub.tick_params(axis='both', colors='black', labelsize=32)
        for spine in ax_sub.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            
        # 在所有子图上添加 y 轴和 x 轴标签
        ax_sub.set_ylabel('Frequency (MHz)', fontsize=32, color='black', labelpad=5)
        ax_sub.set_xlabel('Time (s)', fontsize=32, color='black', labelpad=15)
        
        # 添加子标题
        if i < len(titles):
            ax_sub.set_title(titles[i], fontsize=32, color='black', pad=20)

    plt.savefig(out_path, dpi=100)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    in_path = "/home/cbm/deRFI/results/comparison_report/trt_comp_001_miou0.566.png"
    out_path = "/home/cbm/deRFI/results/comparison_report/trt_comp_001_miou0.566_with_axes.png"
    main(in_path, out_path)
