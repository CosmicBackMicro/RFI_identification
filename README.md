# 射电干扰 (RFI) 深度学习消除项目帮助文档

Disclaimer：这个交接文档由项目上一任负责人CBM编写，部分内容为AI生成，但经过CBM的逐段审核。如果对未提及的内容有疑问，可以联系CBM或查看他的毕业论文，另外Cao et al. 2026, RAA, "RFI Identification and Classification for FAST-GPPS Survey via Transformer Architectures"也可以让你对项目有一个更好的理解，建议先读一下。组内的其他成员知道CBM的联系方式。

本项目利用深度学习语义分割手段，对射电天文数据中的射电频率干扰（RFI）进行识别、掩码与消除。为了你能快速上手，本文档整理了项目的核心概念、环境配置，以及从数据标记到推理的完整工作流。

---

## 0. 环境准备与项目结构

**依赖安装**：  
Python 依赖可以通过 `requirements.txt` 安装：
```bash
pip install -r requirements.txt
```
*此外，项目涉及 C 语言和 OpenMP 的编译，需确保系统中安装了 `gcc`/`g++` 以及 Make 工具。*

**核心目录说明**：
- `src/`：存放所有源代码，包括 C 语言的解析标记程序和 Python 深度学习脚本。
- `Datasets/`：用于存放结构化的训练和验证数据集。你可以在这里找到我制作好的大规模数据集（如 `SynthesizedDataset`）。
- `output/`：C 语言程序处理原始 FITS 数据后的默认输出目录（图像与标记掩码对）。
- `checkpoints/` & `training_logs/`：训练过程中生成的模型权重与 TensorBoard 日志。
- `results/`：验证和评测脚本（如混淆矩阵作图）的输出文件。

---

## 1. 数据标记 (Data Labeling)

本项目第一步是对 PSRFITS 原始数据进行半自动标注。

- **核心程序**：`src/ReadFASTData.c`
- **功能**：读取原始射电数据，利用基于阈值的 RFI 标记算法将干扰特征提取在 mask 上。程序通过 OpenMP 实现了主循环的 CPU 多线程并行，速度极快。（*注意：如果修改主循环逻辑，务必避免线程竞争*）。
- **编译执行**：
  在项目根目录下直接使用 `make`：
  ```bash
  make release  # 或 make turbo 获取最大优化，调试时用 make debug
  ```
- **参数与下采样**：程序的命令行参数我记录在 `.vscode/launch.json` 中，你可以直接在 VSCode 中按 F5 调试。
重要参数： `--binFactorTime` 和 `--blocksPerRead` 可以控制数据的下采样倍数。通常这俩参数要保持相等，这样不同下采样的图像尺寸才能保持一致。

---

## 2. 数据集构造 (Dataset Construction)

`ReadFASTData` 生成的图像为 `.fits` 格式，掩码为 `.png` 格式（目前也支持通过 `visualize_fits` 察看 `.fits` 掩码），它们存放在 `output/`。在交给深度学习模型前，需要结构化。

- **目录结构要求**：
  模型 DataLoader 依赖如下组织形式：
  ```text
  Dataset_Name/
      ├── image/
      │   ├── train/
      │   └── val/
      └── mask/
          ├── train/
          └── val/
  ```
- **常用构建脚本**：
  - `src/split_dataset.py`：自动将 `output/` 里的样本按比例划分并移入上述结构的文件夹（可通过 `--help` 查看用法）。
  - `src/merge_datasets.py`：我一般是把每个原始FITS文件处理成一个单独的数据集结构，用此脚本合并成一个大的数据集结构。
  - **数据可视化**：`src/visualize_fits.py` 是我专门定制的针对本项目FITS数据集的可视化工具，运行它会弹出窗口，根据提示指定数据集目录即可查看。支持开关 mask 叠加、图例、预览不同sigma阈值截断的通道（各有对应的快捷键，详见--help），强烈建议用它进行数据可视化。
  - **数据仿真**：`src/RFISimulator.py` 可用于在训练集中加入仿真 RFI 数据以增强数据多样性，这可以辅助模型学习通用的特征表示。通过--psrfits参数，可以控制输出的仿真数据是分成单个subint存储还是合成一个大的PSRFITS文件。详见--help。

---

## 3. 模型训练与评估 (Model Training & Evaluation)

训练部分基于 PyTorch Lightning 和 Segmentation Models PyTorch 构建，支持多卡及混合精度。

- **核心训练脚本**：
  - `src/UNet.py` & `src/SegFormer.py`：基础开发与测试脚本。
  - `src/UNetOnServer.py` & `src/SegFormerOnServer.py`：专门为 g02 服务器集群（V100S 架构）优化的版本。启用了 DDP 多卡训练、显存限制、梯度裁剪、针对 `block` 等难学类别调整的 He 初始化与分组 AdamW 优化策略。
- **使用方法**：
  调整脚本底部的全局变量（如 Epoch、Batch Size、加载路径），直接运行即可：
  ```bash
  python src/UNetOnServer.py
  ```
- **模型评估与验证**：
  - 训练时 TensorBoard 自动记录。
  - `src/export_metrics.py`：可以解析 TensorBoard 日志，直接在终端打印或导出高精度（3位小数）的 CSV 验证指标表格（精确度、召回率、F1、IoU）。运行示例：
    ```bash
    python src/export_metrics.py /path/to/log_dir
    ```
  - `src/validate_unet.py` & `src/plot_custom_cm.py`：用于加载已训练好的 `.ckpt` 权重，手动跑一次验证集并生成极高精度的混淆矩阵图与 `.npy` 原始频数，非常适合论文作图准备。
  - 传统对比：`src/aoflagger_mask.py`（基于 `parkes-default.lua`）用于生成经典的 AOFlagger 掩码，可用于对比 AI 的准确率。

---

## 4. 模型导出与优化 (Export & Optimization)

为了将庞大的 PyTorch 模型应用于实际的高吞吐量流线，我们需要加速。

- **核心脚本**：`src/exportModel.py`
- **功能**：将训练得到的 `.ckpt` 权重转换为 ONNX 及 TensorRT 引擎。这通常会带来 **2 到 4 倍的推理速度提升**，代价是需要固定推理时的图像尺寸（丢弃动态分辨率）。
- **下一步优化方向（留给你的挑战）**：
  脚本内可以进一步添加对 **INT8 量化** 的支持，这能再提升 2 倍速度并缩减一半显存占用。提醒：INT8 量化需要精选 Calibration 数据集，否则准确率会崩盘。

---

## 5. 模型推理 (Inference)

这是项目的终点，将模型应用于真实观测数据。

- **核心脚本**：`src/AI_RFI.py`
- **功能**：加载优化后的模型（或 ckpt），对新来的 FITS 数据生成消噪后的 mask。
- **高阶操作**：该脚本带有直接在源文件像素上执行重置替换的逻辑选项（默认关闭，属高风险操作）。具体替换的物理意义与阈值逻辑，请参考我的毕业论文。

---

## 6. 数据资产与未来展望

- **超大数据集**：`/home/cbm/deRFI/Datasets/SynthesizedDataset`（g02 上有备份，大约 4 万张图，100GB）。由于是半自动批量标记，个别复杂样本难免有瑕疵，后续可以考虑数据清洗与精制。
- **改进方向建议**：
  1. **数据多样性**：继续完善生成式 RFI 模拟。
  2. **自监督预训练**：这是非常有希望走向巡天级别泛化能力的方向。我已在 g02 上存放了基于 DINO v2 的实验性预训练脚本，目标是通过 Masked AutoEncoder 等技术减少对完美标记掩码的依赖，这个脚本仍有一些 bugs 待修复，留待你继续探索。

祝你在射电数据抗干扰领域取得更出色的进展！