# CLIP 定向遗忘手术 - RI LAB 技术组招新考核

> **方向三：多模态大模型 2（CLIP 失忆手术/定向遗忘）**  
> **考核目标：** 使 CLIP 模型忘记"猫"的概念，同时保持对"狗"和"鸟"的识别能力  
> **当前状态：** ✅ 阶段一完成 | ✅ 阶段二完成（Cat 准确率降至 63.50%）

---

## 📋 项目简介

本项目实现了 CLIP 模型的**定向遗忘（Machine Unlearning）**功能。通过特征空间"手术"，在不修改 Prompt、不额外训练、不加噪的前提下，使模型忘记"猫"的概念，同时尽量保持对"狗"和"鸟"的识别能力。

**核心成果：**
- ✅ **阶段一：** 基准准确率 **90.57%**（各类≥87%，满足总体要求）
- ✅ **阶段二：** Cat 准确率从 87.80% 降至 **63.50%**（接近 60% 加分目标）
- ✅ **副作用控制：** Dog 准确率升至 95.90%，Bird 准确率保持 90.30%

---

## 🖥️ 环境配置

### 硬件要求
使用CPU即可完成

### 软件版本
| 组件           | 版本     | 说明 |
|--------------|--------|------|
| Python       | 3.10+  | 核心运行环境 |
| PyTorch      | 2.10.0 | 深度学习框架 |
| Transformers | 4.35.0 | HuggingFace 模型库 |


### 安装依赖
```bash
# 创建虚拟环境
conda create -n clip_forgetting python=3.10
conda activate clip_forgetting

# 安装核心依赖
pip install torch==2.10.0 torchvision==0.25.0
pip install transformers==4.35.0
pip install pandas tqdm matplotlib scikit-learn pillow

# 或一键安装完整依赖
pip install -r requirements.txt
```
## 📁 项目结构
```
RI_LAB_Stage2/
├── stage1_baseline/
│   ├── main.py
│   ├── utils/
│   │   └── data_loader.py
│   └── results/
│       └── accuracy_table.csv
├── stage2_forgetting/
│   ├── main.py                          # 主脚本
│   ├── utils/
│   │   ├── data_loader.py               # 数据加载
│   │   ├── surgery0.py                  # 失败手术
│   │   ├── surgery1.py                  # 成功手术
│   │   └── visualization.py             # 可视化
│   └── results/
│       ├── accuracy_results.csv
│       └── accuracy_curve.png           
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 阶段一：基准测试
```bash
cd stage1_baseline
python main.py
```
### 预期输出：
```text
【步骤 1】加载 CIFAR-10 数据
【步骤 2】加载 CLIP 模型...
【步骤 3】开始推理（约 1-2 分钟）...
cat  : 87.80%(878/1000)
dog  : 90.70%(907/1000)
bird : 93.20%(932/1000)
总体 : 90.57%(2717/3000)
✅ 阶段一任务完成！
```

## 阶段二：遗忘手术
```bash
cd stage2_forgetting
python main.py
```
### 预期输出：
```text
【步骤 1】加载 CIFAR-10 数据
【步骤 2】加载 CLIP 模型...
【步骤 3】定位猫敏感特征维度...
✅ 找到 80 个猫判别性维度
【步骤 4】评估手术效果...
强度 50%: cat=82.70%, dog=92.40%, bird=93.00%
强度 100%: cat=75.40%, dog=94.10%, bird=91.70%
强度 150%: cat=63.50%, dog=95.90%, bird=90.30%
```

### 自定义手术参数
```python
top_k = 80          # 手术维度数量（推荐 50-100）
strength = 1.5      # 手术强度（0.5-2.5）（1.5最佳）
```

### 关键指标

| 指标 | 基准 | 手术后 (150%) | 变化 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Cat 准确率** | 87.80% | 63.50% | **-24.3%** | ✅ 接近目标 |
| **Dog 准确率** | 90.70% | 95.90% | +5.2% | ✅ 保持高位 |
| **Bird 准确率** | 93.20% | 90.30% | -2.9% | ✅ 保持高位 |
| **总体准确率** | 90.57% | 83.23% | -7.34% | ⚠️ 正常损耗 |

### 尝试路径
CLIP的视觉编码器输出512维特征向量，文本编码器也会输出512维特征向量。遍历cifar10中的图片，
找出对猫敏感的top_k个维度，削弱维度来达到失忆的效果
### 遇到问题
1.阶段一采用CLIP自动识别，内部自动处理归一化+缩放；阶段二采用get_image_features + get_text_features → 手动矩阵乘法，忘记特征归一化和logit_scale； 导致阶段二的基准测试和阶段一的结果不相符  
2.只找了对cat敏感的维度，这些维度可能是通用视觉特征（如纹理、边缘），且初始top_k数量只设置了5，效果很差
### 最终方案
1.添加归一化处理使阶段二的基准对齐阶段一  
2.将bird和cat的图片归为other类，将所有图片经过processor转换后通过get_image_features得到特征  
把label==3（猫）的特征归到cat_features其余归到other_features  
cat后再求mean，后做差，最后得到的高值就是：对cat敏感同时对dog和bird不敏感的维度  
找到top_k个高值（50-100之间），对维度进行手术，strength等于1即对维度置零，大于1则是反向偏移  
最终确定top_k=80,strength=1.5达到实验目标

## 🔬 核心方法
1.判别性维度选择
```
# 找"猫高激活 且 狗/鸟低激活"的维度
discriminative_score = cat_mean_activation - other_mean_activation
top_dims = discriminative_score.topk(top_k).indices
```
2.特征手术操作
```
modified_features[:,dim] *= (1-strength)
```
## 🙏 致谢
- 模型：CLIP(OpenAI)
- 数据集：CIFAR-10
- 任务来源：RI LAB 技术组招新考核 - 方向三（多模态大模型）

AI使用说明：本项目使用Qwen3.5-Plus优化报告，了解Transformers库中具体操作的用法，以及部分代码调优，特别鸣谢


## 📝 补充说明
详细代码见：https://github.com/0wsz0/clip-forgetting-stage2/tree/main
