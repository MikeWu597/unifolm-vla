# UnifoLM-VLA vast.ai 部署指南

## 概述

本指南介绍如何在 [vast.ai](https://vast.ai) 平台租用 GPU 实例，复现 UnifoLM-VLA 在 LIBERO 仿真环境中的推理评估。

**预期总耗时**: ~1-2 小时（含环境搭建、权重下载、小规模评估）

**预期花费**: ~$1-3（RTX 4090 约 $0.3/h × 3-5 小时，含调试余量）

---

## 1. 租用 GPU 实例

### 1.1 推荐配置

| 要求 | 推荐 | 最低 |
|------|------|------|
| GPU | RTX 4090 / A6000 | RTX 3090 |
| 显存 | 24 GB+ | 20 GB |
| 系统盘 | 100 GB+ | 60 GB |
| CUDA | 12.4 | 12.x |
| 镜像 | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | Ubuntu 22.04 + 手动安装 |

### 1.2 租用步骤

1. 打开 [vast.ai/console/create](https://vast.ai/console/create/)
2. **Filters** 设置:
   - GPU RAM: ≥ 24 GB
   - CUDA: ≥ 12.4
   - Storage: ≥ 80 GB
   - 建议勾选 "Unverified" 以获取更低价格（但优先选高评分 host）
3. **Image**: 选择 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` 或直接 Ubuntu 22.04
4. 点击 **RENT**，等待实例启动（通常 1-3 分钟）
5. 实例就绪后，点击 **Connect** → 复制 SSH 命令

### 1.3 SSH 连接

```bash
ssh -p <PORT> root@<HOST>
# 或使用 vast.ai 提供的 Jupyter / Web Terminal
```

---

## 2. 环境搭建

### 2.1 一键安装（推荐）

将本仓库的三个文件上传到实例后执行:

```bash
# 上传文件
scp -P <PORT> setup_vastai.sh download_weights.py run_libero_eval.sh root@<HOST>:/root/

# SSH 进入实例，执行安装
ssh -p <PORT> root@<HOST>
bash /root/setup_vastai.sh
```

安装过程约 15-30 分钟，其中 **flash-attn 编译最耗时**（5-15 分钟）。

### 2.2 常见问题

| 问题 | 解决方案 |
|------|----------|
| flash-attn 编译失败 | 尝试 `pip install flash-attn --no-build-isolation`（不加版本号，装最新版）|
| CUDA 版本不匹配 | vast.ai 重租实例，选择正确的 CUDA 镜像 |
| 磁盘空间不足 | 权重总约 30GB，确保 /root 或 /workspace 有 80GB+ |
| git clone 太慢 | `git clone --depth 1` 浅克隆 |
| conda 不可用 | 使用镜像自带的 conda，或改用 venv + pip |

---

## 3. 下载模型权重

```bash
conda activate unifolm-vla
cd /root/unifolm-vla

# 国内网络可用镜像加速:
HF_ENDPOINT=https://hf-mirror.com python download_weights.py --output-dir /root/models

# 海外直连:
python download_weights.py --output-dir /root/models
```

**需要下载的模型**:
- `UnifoLM-VLM-Base` (~16 GB) — Qwen2.5-VL 基础模型
- `UnifoLM-VLA-LIBERO` (~16 GB) — LIBERO 微调 checkpoint

下载时间取决于网络，通常 10-30 分钟。

### 3.1 验证权重下载

下载完成后，确认文件结构正确:

```bash
# VLM-Base 应包含 HuggingFace 标准文件
ls /root/models/UnifoLM-VLM-Base/
# 应有: config.json, model.safetensors, tokenizer.json, ...

# VLA-LIBERO 应包含 checkpoint + 配置文件
ls /root/models/UnifoLM-VLA-LIBERO/
# 应有: config.yaml, dataset_statistics.json, checkpoints/pytorch_model.pt
```

> **注意**: 如果 HuggingFace 仓库中的文件结构与预期不同，需要调整 `run_libero_eval.sh` 中的路径。
> 可以先运行 `find /root/models -name "*.pt"` 找到实际的 checkpoint 文件。

---

## 4. 运行评估

### 4.1 修改配置

编辑 `run_libero_eval.sh` 中的路径:

```bash
vim /root/unifolm-vla/run_libero_eval.sh

# 关键变量:
# VLM_PRETRAINED_PATH=/root/models/UnifoLM-VLM-Base
# VLA_CHECKPOINT=/root/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt
# TASK_SUITE=libero_spatial     # 从最简单的开始
# NUM_TRIALS=5                   # 先跑 5 个 episode 验证
```

### 4.2 快速验证（单任务套件，5 episodes）

```bash
cd /root/unifolm-vla
bash run_libero_eval.sh
```

如果一切正常，5 个 episode 约需 5-10 分钟。成功时输出类似:

```
Task: put the black bowl on the plate
Starting episode 1...
Success: True
# successes: 5 (100.0%)
Current task success rate: 1.0
```

### 4.3 正式评估

验证通过后，修改配置跑完整评估:

```bash
# 编辑 run_libero_eval.sh:
#   TASK_SUITE=libero_spatial
#   NUM_TRIALS=50

bash run_libero_eval.sh
```

五个任务套件的完整评估（每个 50 episodes × 5 suites）约需 3-6 小时。

---

## 5. 任务套件说明

| 套件 | 任务数 | 每任务最大步数 | 预期成功率 | 说明 |
|------|--------|---------------|-----------|------|
| `libero_spatial` | 10 | 220 | ~99% | 空间关系：左/右/上/下 |
| `libero_object` | 10 | 280 | ~100% | 物体类型：碗/盘子/杯子 |
| `libero_goal` | 10 | 300 | ~99.4% | 任务目标：拿起/放入 |
| `libero_10` | 10 | 520 | ~96.2% | 长周期组合任务 |
| `libero_90` | 90 | 400 | ~ | 全量评估 |

论文报告的 UnifoLM-VLA 在 LIBERO 上的平均分: **98.7** (Spatial: 99.0, Object: 100, Goal: 99.4, Long: 96.2)

---

## 6. 目录结构（运行后）

```
/root/
├── models/
│   ├── UnifoLM-VLM-Base/          # VLM 基础模型 (~16GB)
│   └── UnifoLM-VLA-LIBERO/        # VLA checkpoint (~16GB)
├── LIBERO/                         # LIBERO 仿真环境
└── unifolm-vla/                    # UnifoLM-VLA 代码
    ├── setup_vastai.sh
    ├── download_weights.py
    ├── run_libero_eval.sh
    ├── results/                    # 评估结果 (视频 + 日志)
    │   └── libero_spatial/
    │       └── 20260101_120000/
    └── experiments/
        └── LIBERO/
            ├── eval_libero.py      # 评估入口
            └── unifolm_vla_inference.py
```

---

## 7. 架构说明

```
┌──────────────────────────────────────────┐
│ 输入: LIBERO 渲染图像 + 任务文本           │
│   - agentview (224×224)                  │
│   - wrist camera (224×224)               │
│   - robot state (7D EEF + quat)          │
├──────────────────────────────────────────┤
│ Qwen2.5-VL-7B-Instruct (bf16)            │
│   → 编码图像 + 文本 → hidden_states       │
│   ~14 GB VRAM                            │
├──────────────────────────────────────────┤
│ FlowmatchingActionHead (fp32)            │
│   → DiT 扩散去噪 → 预测动作序列           │
│   ~2 GB VRAM                             │
├──────────────────────────────────────────┤
│ 输出: 7D 动作 (xyz + roll/pitch/yaw + grip)│
│   经过去归一化 → LIBERO 环境执行           │
└──────────────────────────────────────────┘
```

**推理流程:**
1. 维持一个 `window_size=2` 的观测滑动窗口
2. 每 `NUM_ACTIONS_CHUNK=8` 步预测一次（预测 8 步动作序列）
3. 动作序列逐步出队执行，序列耗尽后重新预测
4. 动作反归一化：q01/q99 百分位归一化 → 实际关节范围

---

## 8. 故障排除

### 显存不足 (OOM)
```
torch.cuda.OutOfMemoryError
```
- 确认使用 RTX 4090 (24GB) 或更高
- 关闭其他进程释放显存
- `export CUDA_VISIBLE_DEVICES=0` 只使用单卡

### Checkpoint 加载失败
```
RuntimeError: Missing keys in state_dict: {...}
```
- 检查 `VLA_CHECKPOINT` 路径是否正确
- 确认 `VLM_PRETRAINED_PATH` 指向正确的 VLM-Base 目录
- 确认 `config.yaml` 和 `dataset_statistics.json` 在 checkpoint 的上两级目录

### LIBERO 环境错误
```
ImportError: No module named 'libero'
```
- 确认 LIBERO 已安装: `pip install -e /root/LIBERO`
- 确认 `PYTHONPATH` 包含 LIBERO 路径
- 检查 mujoco 安装: `python -c "import mujoco"`

### constants.py 检测到错误的平台
```
Using G1_EE_6D constants:
  NUM_ACTIONS_CHUNK = 25
  ACTION_DIM = 23
```
- `eval_libero.py` 的 CLI 参数中包含 `libero` 关键词，constants.py 会自动检测
- 如果检测到错误平台，在 CLI 参数中添加 `--help` 确认参数名正确

---

## 9. 清理

评估完成后记得销毁实例避免持续计费:

```bash
# 在 vast.ai 控制台 → Instances → Destroy
# 或 CLI:
vastai destroy instance <INSTANCE_ID>
```

---

## 参考

- UnifoLM-VLA GitHub: https://github.com/unitreerobotics/unifolm-vla
- UnifoLM-VLA 论文引用: `@misc{unifolm-vla-0, author={Unitree}, title={UnifoLM-VLA-0}, year={2026}}`
- LIBERO Benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO
- HuggingFace 模型集: https://huggingface.co/collections/unitreerobotics/unifolm-wma-0
