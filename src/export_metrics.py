import os
import sys
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import glob

def get_latest_event_file(log_dir):
    """获取最新的 TensorBoard event 文件"""
    # 如果是目录，递归查找 event 文件
    if os.path.isdir(log_dir):
        event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
    else:
        # 如果传入的是具体的文件路径，直接用
        event_files = [log_dir] if "events.out.tfevents" in log_dir else []
        
    if not event_files:
        return None
    # 按修改时间排序
    event_files.sort(key=os.path.getmtime, reverse=True)
    return event_files[0]

def extract_last_epoch_metrics(log_dir):
    event_file = get_latest_event_file(log_dir)
    if not event_file:
        print(f"Error: No event files found in {log_dir}")
        return

    print(f"Reading from: {event_file}")
    print(f"📄 [Event File] Full path: {os.path.abspath(event_file)}")
    
    # 获取所属的目录名，用于提示用户正在读取哪个 version
    version_dir = os.path.basename(os.path.dirname(event_file))
    print(f"📊 [Target Log] Detected training log version: {version_dir}")
    
    # 这里的 size_guidance 设置是为了加载数据的粒度，标量设置较大的值确保不被截断
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = ea.Tags()['scalars']
    
    # 我们关心的前缀 (适配 SegFormer 和新的 UNet 脚本)
    # SegFormer 可能使用: val_iou_cls_x_name
    # UNet 使用: val_iou_name
    target_prefixes = ['val_f1_', 'val_iou_', 'val_precision_', 'val_recall_']
    
    results = []
    
    for tag in tags:
        match_prefix = None
        for p in target_prefixes:
            if tag.startswith(p):
                match_prefix = p
                break
        
        if match_prefix:
            events = ea.Scalars(tag)
            if events:
                last_event = events[-1] # 获取最后一个 epoch
                
                # 尝试解析类别名称
                # 情况 1: val_f1_cls_3_point (SegFormer)
                # 情况 2: val_f1_point (UNet)
                if '_cls_' in tag:
                    parts = tag.split('_')
                    metric_type = parts[1] # f1, iou...
                    class_name = "_".join(parts[4:])
                else:
                    # 处理 val_f1_point 这种格式
                    metric_type = match_prefix.split('_')[1] # f1, iou...
                    class_name = tag.replace(match_prefix, "")
                
                # 统一指标名称缩写
                label_map = {
                    'f1': 'F1',
                    'iou': 'Iou',
                    'precision': 'Pre',
                    'recall': 'Rec'
                }
                
                results.append({
                    'Class': class_name,
                    'Metric': label_map.get(metric_type, metric_type.capitalize()),
                    'Value': float(last_event.value),
                    'Step': last_event.step,
                    'Epoch': last_event.step
                })

    if not results:
        print("No matching metrics found. Make sure validation has completed at least one epoch.")
        return

    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='Class', columns='Metric', values='Value')
    
    desired_order = ['Iou', 'Pre', 'Rec', 'F1']
    available_cols = [c for c in desired_order if c in pivot_df.columns]
    pivot_df = pivot_df[available_cols]

    # 设置 pandas 的显示精度并在打印/保存前保留 3 位小数
    pd.set_option('display.precision', 3)
    pivot_df = pivot_df.round(3)

    print("\n" + "="*50)
    print("      LAST EPOCH VALIDATION METRICS PER CLASS")
    print("="*50)
    print(pivot_df)
    print("="*50)

    # 保存 pivot_df 为 CSV
    if os.path.isdir(log_dir):
        output_csv = os.path.join(log_dir, "last_epoch_metrics.csv")
    else:
        # 如果传入的是具体文件，保存在文件同级目录下
        output_csv = os.path.join(os.path.dirname(log_dir), "last_epoch_metrics.csv")
        
    pivot_df.to_csv(output_csv)
    print(f"Saved metrics to {output_csv}")
    
    # 可选：保存到 CSV
    # pivot_df.to_csv("last_epoch_metrics.csv")
    # print("Saved to last_epoch_metrics.csv")

if __name__ == "__main__":
    # 支持从命令行传入参数，例如: 
    # python src/export_metrics.py 137
    # python src/export_metrics.py version_137
    # python src/export_metrics.py /path/to/logs
    
    BASE_LOG_DIR = "/home/cbm/deRFI/training_logs"
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # 如果传入的是数字，拼接成 version_x
        if arg.isdigit():
            LOG_DIR = os.path.join(BASE_LOG_DIR, f"version_{arg}")
        # 如果传入的是 version_x 字符串
        elif arg.startswith("version_"):
            LOG_DIR = os.path.join(BASE_LOG_DIR, arg)
        # 如果是绝对路径
        elif os.path.isabs(arg):
            LOG_DIR = arg
        else:
            # 尝试在当前目录下找
            LOG_DIR = os.path.abspath(arg)
    else:
        # 默认使用最近的一个 version（如果有）
        if os.path.exists(BASE_LOG_DIR):
            versions = [d for d in os.listdir(BASE_LOG_DIR) if d.startswith("version_")]
            if versions:
                v_sorted = sorted(versions, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
                LOG_DIR = os.path.join(BASE_LOG_DIR, v_sorted[-1])
                print(f"No version provided, using latest: {v_sorted[-1]}")
            else:
                LOG_DIR = BASE_LOG_DIR
        else:
            LOG_DIR = "."
        
    if not os.path.exists(LOG_DIR):
        print(f"❌ 路径不存在: {LOG_DIR}")
        # 尝试列出最新 5 个版本
        if os.path.exists(BASE_LOG_DIR):
            versions = [d for d in os.listdir(BASE_LOG_DIR) if d.startswith("version_")]
            if versions:
                v_sorted = sorted(versions, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
                print(f"提示：最近的 version 包含: {v_sorted[-5:]}")
    else:
        extract_last_epoch_metrics(LOG_DIR)
