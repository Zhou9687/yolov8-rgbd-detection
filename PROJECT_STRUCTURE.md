# RGBD YOLOv8 项目结构文档

## 📁 项目目录树

```
D:\ProjectCode\PyCharm\ultralytics-main/
│
├── 📄 README.md                          # 项目主说明文档
├── 📄 RGBD_PROBLEM_ANALYSIS.md          # ⭐ 问题分析报告
├── 📄 RGBD_SOLUTION_GUIDE.md            # ⭐ 解决方案指南
├── 📄 RGBD_TRAINING_GUIDE.md            # ⭐ RGBD训练指南
├── 📄 PROJECT_STRUCTURE.md              # ⭐ 本文档 - 项目结构说明
│
├── 📄 pyproject.toml                     # Python项目配置
├── 📄 requirements.txt                   # Python依赖列表
├── 📄 LICENSE                            # 开源协议
├── 📄 CONTRIBUTING.md                    # 贡献指南
│
├── 🔧 train_rgbd_direct.py              # ⭐ RGBD训练主脚本
├── 🔧 check_data_loading.py             # 数据加载测试脚本
├── 🔧 preprocess_rgbd.py                # RGBD预处理脚本
│
├── 🏋️ yolov8n.pt                        # YOLOv8n 3通道预训练权重
├── 🏋️ yolov8_4ch_direct.pt              # ⭐ YOLOv8 4通道预训练权重
├── 🏋️ yolo11n.pt                        # YOLO11n 预训练权重
│
├── 📂 ultralytics/                       # ⭐ Ultralytics核心库（已修改）
│   ├── 📄 __init__.py
│   │
│   ├── 📂 cfg/                          # 配置文件
│   │   └── 📂 models/
│   │       └── 📂 v8/
│   │           ├── 📄 yolov8.yaml       # 标准YOLOv8配置
│   │           └── 📄 yolov8-rgbd.yaml  # ⭐ RGBD 4通道配置
│   │
│   ├── 📂 data/                         # ⭐ 数据处理模块（核心修改）
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base.py                   # 数据集基类
│   │   ├── 📄 dataset.py                # ⭐ 数据加载器（已修改load_image）
│   │   ├── 📄 augment.py                # 数据增强
│   │   ├── 📄 build.py                  # 数据加载器构建
│   │   └── 📄 utils.py                  # 数据处理工具
│   │
│   ├── 📂 engine/                       # 训练引擎
│   │   ├── 📄 __init__.py
│   │   ├── 📄 model.py                  # 模型接口
│   │   ├── 📄 trainer.py                # 训练器
│   │   ├── 📄 validator.py              # 验证器
│   │   └── 📄 predictor.py              # 推理器
│   │
│   ├── 📂 nn/                           # 神经网络模块
│   │   ├── 📄 __init__.py
│   │   ├── 📂 modules/                  # 网络层
│   │   │   ├── 📄 conv.py              # 卷积层
│   │   │   ├── 📄 block.py             # 网络块
│   │   │   └── 📄 head.py              # 检测头
│   │   └── 📂 tasks.py                  # 任务定义
│   │
│   ├── 📂 models/                       # 模型定义
│   │   └── 📂 yolo/
│   │       ├── 📄 detect.py            # 检测模型
│   │       └── 📄 model.py             # YOLO模型
│   │
│   ├── 📂 utils/                        # 工具函数
│   │   ├── 📄 __init__.py
│   │   ├── 📄 ops.py                   # 操作函数
│   │   ├── 📄 checks.py                # 检查函数
│   │   └── 📄 torch_utils.py           # PyTorch工具
│   │
│   └── 📂 assets/                       # 资源文件
│
├── 📂 scripts/                          # ⭐ 辅助脚本
│   ├── 📄 prepare_4ch_weights.py        # ⭐ 创建4通道预训练权重
│   ├── 📄 fuse_rgb_depth.py            # RGB+Depth融合脚本
│   ├── 📄 check_pt.py                  # 模型检查脚本
│   ├── 📄 check_is_four_path.py        # 4通道验证脚本
│   └── 📄 train_4ch_fixed.py           # 4通道训练脚本
│
├── 📂 datasets/                         # ⭐ 数据集目录
│   ├── 📂 tennis-yolo/                  # ⭐ Tennis Ball RGBD数据集
│   │   ├── 📄 tennis-yolo.yaml         # ⭐ 数据集配置（rgbd:true）
│   │   ├── 📂 images/
│   │   │   ├── 📂 train/               # 训练图像（4通道PNG）
│   │   │   │   ├── 🖼️ image_001.png    # 480×640×4 RGBA
│   │   │   │   ├── 🖼️ image_002.png
│   │   │   │   └── ... (60张)
│   │   │   └── 📂 val/                 # 验证图像
│   │   │       ├── 🖼️ val_001.png
│   │   │       └── ... (30张)
│   │   └── 📂 labels/
│   │       ├── 📂 train/               # 训练标注
│   │       │   ├── 📄 image_001.txt    # YOLO格式
│   │       │   └── ...
│   │       └── 📂 val/                 # 验证标注
│   │           └── ...
│   │
│   ├── 📂 tennis-rgbd/                  # 原始RGBD数据集
│   └── 📂 tennis_path/                  # 分离的RGB和Depth
│       ├── 📂 Color/
│       └── 📂 Depth/
│
├── 📂 runs/                             # ⭐ 训练结果目录
│   └── 📂 detect/
│       ├── 📂 train16/                  # 早期训练（3通道）
│       │   └── 📂 weights/
│       │       ├── 🏋️ best.pt          # ❌ 3通道模型
│       │       └── 🏋️ last.pt
│       │
│       └── 📂 train_rgbd_python_api36/  # ⭐ 最终成功训练（4通道）
│           ├── 📄 args.yaml            # 训练参数
│           ├── 📄 results.csv          # 训练指标
│           ├── 📊 results.png          # 指标曲线
│           ├── 📊 confusion_matrix.png # 混淆矩阵
│           ├── 📊 BoxPR_curve.png      # PR曲线
│           ├── 📊 BoxF1_curve.png      # F1曲线
│           ├── 🖼️ labels.jpg           # 标签分布
│           ├── 🖼️ train_batch*.jpg     # 训练样本
│           ├── 🖼️ val_batch*_pred.jpg  # 验证预测
│           └── 📂 weights/
│               ├── 🏋️ best.pt          # ⭐ 4通道最佳模型
│               └── 🏋️ last.pt          # ⭐ 4通道最后模型
│
├── 📂 tests/                            # 测试脚本
│   ├── 📄 test_cuda.py
│   ├── 📄 test_engine.py
│   └── 📄 test_python.py
│
├── 📂 examples/                         # 示例代码
│   ├── 📂 YOLOv8-ONNXRuntime/
│   ├── 📂 YOLOv8-CPP-Inference/
│   └── ...
│
└── 📂 docs/                             # 文档
    ├── 📄 build_docs.py
    └── 📂 en/
        └── 📄 index.md

```

---

## 🎯 核心文件说明

### 1. 修改的核心文件 ⚙️

#### `ultralytics/data/dataset.py`

**修改位置**: 第 92-191 行  
**修改内容**: `load_image()` 方法

**关键修改点**:

```python
# 修改前
im = cv2.imread(im_path)  # 只读取3通道
return im, (h, w)  # 返回2个值

# 修改后
im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)  # 读取所有通道
if im.shape[2] == 4:  # 检测4通道
    b, g, r, a = cv2.split(im)
    im = cv2.merge([r, g, b, a])  # BGRA → RGBA
return im, (h, w), im.shape[:2]  # 返回3个值
```

**影响范围**:

- ✅ 支持4通道PNG图像加载
- ✅ 正确的颜色空间转换
- ✅ 兼容原有3通道模式

---

#### `ultralytics/cfg/models/v8/yolov8-rgbd.yaml`

**文件类型**: 模型配置文件  
**关键配置**:

```yaml
# YOLOv8-RGBD 4-channel model
nc: 1 # number of classes
depth_multiple: 0.33
width_multiple: 0.25
ch: 4 # ⭐ 输入通道数改为4

# Backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2  ← 第一层接收4通道输入
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  # ... 其余层保持不变
```

**作用**:

- 定义4通道输入的YOLOv8架构
- 第一层卷积从 `Conv(3, 16, ...)` 变为 `Conv(4, 16, ...)`

---

### 2. 新增的核心文件 ✨

#### `scripts/prepare_4ch_weights.py`

**功能**: 创建4通道预训练权重  
**输入**: `yolov8n.pt` (3通道)  
**输出**: `yolov8_4ch_direct.pt` (4通道)

**转换逻辑**:

```python
# 读取3通道权重 [16, 3, 3, 3]
weight_3ch = state_dict["model.0.conv.weight"]

# 创建4通道权重 [16, 4, 3, 3]
weight_4ch = torch.zeros(16, 4, 3, 3)
weight_4ch[:, :3, :, :] = weight_3ch  # 复制RGB
weight_4ch[:, 3, :, :] = torch.randn(16, 3, 3) * 0.01  # 初始化深度通道
```

---

#### `train_rgbd_direct.py`

**功能**: RGBD模型训练主脚本  
**类型**: Python API训练脚本

**关键配置**:

```python
model.train(
    data="datasets/tennis-yolo/tennis-yolo.yaml",
    epochs=100,
    batch=4,
    workers=0,  # ⭐ Windows多进程修复
    amp=False,  # ⭐ 禁用AMP检查
    mosaic=0.0,  # ⭐ 禁用Mosaic
    mixup=0.0,  # ⭐ 禁用Mixup
    copy_paste=0.0,  # ⭐ 禁用Copy-Paste
)
```

---

#### `datasets/tennis-yolo/tennis-yolo.yaml`

**功能**: 数据集配置文件  
**关键配置**:

```yaml
path: D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo
train: images/train
val: images/val

nc: 1
names:
  0: tennis_ball

rgbd: true # ⭐ 启用RGBD模式
channels: 4 # ⭐ 4通道输入
```

---

### 3. 输出文件 📊

#### 训练结果目录结构

```
runs/detect/train_rgbd_python_api36/
├── args.yaml                   # 训练参数备份
├── results.csv                 # 训练指标（CSV格式）
│   ├── epoch
│   ├── train/box_loss
│   ├── train/cls_loss
│   ├── metrics/mAP50
│   └── metrics/mAP50-95
│
├── results.png                 # 训练曲线图
├── confusion_matrix.png        # 混淆矩阵
├── confusion_matrix_normalized.png
├── BoxPR_curve.png            # Precision-Recall曲线
├── BoxP_curve.png             # Precision曲线
├── BoxR_curve.png             # Recall曲线
├── BoxF1_curve.png            # F1分数曲线
│
├── labels.jpg                  # 标签分布可视化
├── labels_correlogram.jpg      # 标签相关图
│
├── train_batch0.jpg           # 训练批次可视化
├── train_batch1.jpg
├── train_batch2.jpg
├── train_batch1350.jpg        # 最后几个批次
├── train_batch1351.jpg
├── train_batch1352.jpg
│
├── val_batch0_labels.jpg      # 验证集标签
├── val_batch0_pred.jpg        # 验证集预测
├── val_batch1_labels.jpg
├── val_batch1_pred.jpg
├── val_batch2_labels.jpg
└── val_batch2_pred.jpg

└── weights/
    ├── best.pt                # ⭐ 最佳模型（4通道）
    └── last.pt                # ⭐ 最后epoch模型（4通道）
```

---

## 📦 数据集格式规范

### RGBD图像格式

#### 单文件4通道PNG（推荐）✅

```
文件名: image_001.png
格式: PNG
通道: 4 (RGBA)
尺寸: 480×640×4
数据类型: uint8

通道分配:
- R (通道0): 红色
- G (通道1): 绿色
- B (通道2): 蓝色
- A (通道3): 深度信息（0-255归一化）
```

**读取方法**:

```python
import cv2

img = cv2.imread("image_001.png", cv2.IMREAD_UNCHANGED)
# img.shape = (480, 640, 4)
rgb = img[:, :, :3]  # RGB通道
depth = img[:, :, 3]  # 深度通道
```

#### 分离文件格式（备选）

```
RGB图像: image_001.png     # 3通道彩色图
深度图:  image_001_d.png   # 单通道深度图
```

### YOLO标注格式

```
文件名: image_001.txt
格式: 每行一个目标

<class_id> <x_center> <y_center> <width> <height>

示例:
0 0.5234 0.6128 0.1234 0.0987

说明:
- class_id: 类别ID（从0开始）
- x_center, y_center: 边界框中心点（归一化到0-1）
- width, height: 边界框宽高（归一化到0-1）
```

---

## 🔧 环境配置

### Conda环境

```yaml
名称: yolov8
Python: 3.10.18

主要依赖:
├── torch==2.8.0+cu128          # PyTorch with CUDA
├── torchvision==0.19.0+cu128
├── ultralytics==8.3.176        # 本地修改版
├── opencv-python==4.10.0.84
├── numpy==1.26.4
├── pandas==2.2.3
├── matplotlib==3.9.3
├── pyyaml==6.0.2
└── tqdm==4.67.1
```

### 硬件要求

```
GPU: NVIDIA GeForce RTX 5070 Laptop (8GB VRAM)
CPU: 支持多核处理
RAM: 16GB+
磁盘: 20GB+ 可用空间
操作系统: Windows 11 / Linux
```

---

## 🚀 快速开始指南

### 1. 环境准备

```bash
# 激活conda环境
conda activate yolov8

# 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 创建4通道预训练权重

```bash
cd D:/ProjectCode/PyCharm/ultralytics-main
python scripts/prepare_4ch_weights.py
```

### 3. 验证数据集

```bash
# 检查图像通道数
python -c "
import cv2
img = cv2.imread('datasets/tennis-yolo/images/train/image_001.png',
                 cv2.IMREAD_UNCHANGED)
print(f'Image shape: {img.shape}')
"
```

### 4. 开始训练

```bash
python train_rgbd_direct.py
```

### 5. 查看结果

```bash
# 检查模型通道数
python -c "
import torch
model = torch.load('runs/detect/train_rgbd_python_api36/weights/best.pt',
                   weights_only=False)
ch = model['model'].model[0].conv.weight.shape[1]
print(f'Model channels: {ch}')
"

# 查看训练指标
python -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_rgbd_python_api36/results.csv')
print(df[['epoch', 'metrics/mAP50', 'metrics/mAP50-95']].tail(10))
"
```

---

## 📊 项目统计信息

### 代码统计

```
修改文件数量: 3个核心文件
新增文件数量: 7个脚本和文档
代码行数:
├── dataset.py 修改: ~100行
├── prepare_4ch_weights.py: ~80行
├── train_rgbd_direct.py: ~70行
└── 文档: ~3000行

总计: ~3250行代码和文档
```

### 训练统计

```
数据集规模:
├── 训练图像: 60张
├── 验证图像: 30张
├── 总标注框: 360个
└── 类别数: 1 (tennis_ball)

训练配置:
├── Epochs: 100
├── Batch size: 4
├── Image size: 640×640
├── 训练时长: ~1.5小时
└── GPU显存: ~1.2GB
```

### 文件大小

```
预训练权重:
├── yolov8n.pt:          6.3 MB (3通道)
└── yolov8_4ch_direct.pt: 6.4 MB (4通道)

训练后权重:
├── best.pt:             6.5 MB
└── last.pt:             6.5 MB

数据集:
└── tennis-yolo:         ~50 MB (90张图像+标注)
```

---

## 🔍 关键路径速查

### 训练相关

```bash
# 训练脚本
./train_rgbd_direct.py

# 预训练权重
./yolov8_4ch_direct.pt

# 数据集配置
./datasets/tennis-yolo/tennis-yolo.yaml

# 模型配置
./ultralytics/cfg/models/v8/yolov8-rgbd.yaml
```

### 结果查看

```bash
# 最佳模型
./runs/detect/train_rgbd_python_api36/weights/best.pt

# 训练指标
./runs/detect/train_rgbd_python_api36/results.csv

# 可视化结果
./runs/detect/train_rgbd_python_api36/results.png
./runs/detect/train_rgbd_python_api36/confusion_matrix.png
```

### 文档

```bash
# 问题分析
./RGBD_PROBLEM_ANALYSIS.md

# 解决方案
./RGBD_SOLUTION_GUIDE.md

# 训练指南
./RGBD_TRAINING_GUIDE.md

# 项目结构（本文档）
./PROJECT_STRUCTURE.md
```

---

## 🛠️ 维护和扩展

### 添加新数据集

1. 准备4通道RGBD图像（PNG格式）
2. 创建YOLO格式标注文件
3. 编写数据集YAML配置（参考tennis-yolo.yaml）
4. 设置 `rgbd: true` 和 `channels: 4`

### 修改模型架构

1. 复制 `yolov8-rgbd.yaml`
2. 修改 backbone/head 结构
3. 保持第一层 `ch: 4`
4. 重新生成4通道预训练权重

### 导出模型

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train_rgbd_python_api36/weights/best.pt")

# 导出ONNX
model.export(format="onnx", imgsz=640)

# 导出TensorRT
model.export(format="engine", imgsz=640)
```

---

## 📞 支持和反馈

### 问题报告

如果遇到问题，请提供：

1. 错误信息和堆栈跟踪
2. 环境信息（Python版本、PyTorch版本、GPU型号）
3. 数据集格式示例
4. 复现步骤

### 贡献指南

欢迎提交：

- 🐛 Bug修复
- ✨ 新功能
- 📚 文档改进
- 🧪 测试用例

---

## 📝 版本历史

### v1.0 (2025-11-01)

- ✅ 初始版本
- ✅ 支持4通道RGBD训练
- ✅ 修复所有已知问题
- ✅ 完整文档

---

## 📄 许可证

本项目基于 Ultralytics AGPL-3.0 许可证。详见 LICENSE 文件。

---

**文档维护**: M-Sir-zhou  
**最后更新**: 2025年11月1日  
**项目仓库**: yolov8-rgbd-detection
