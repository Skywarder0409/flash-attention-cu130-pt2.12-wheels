# Flash Attention 2 Wheels for CUDA 13.0 & PyTorch 2.12 (Nightly)

[**🇺🇸 English**](#english) | [**🇨🇳 中文**](#chinese)

---

<a id="english"></a>
## 🇺🇸 English

Pre-built `flash-attn` wheels compiled specifically for **NVIDIA RTX 5090 (Blackwell Architecture, Compute Capability 12.0)** on Linux.

### Motivation

Compiling `flash-attn` from source on newer architectures like Blackwell (RTX 5090) paired with CUDA 13.0 presents significant friction:
1. Local compilation typically exceeds 1 hour.
2. High failure rate due to missing build dependencies, GCC version mismatches, or incompatibility with PyTorch nightly builds.

This repository provides pre-built `.whl` files to bypass the compilation phase entirely. It enables immediate drop-in installation of Flash Attention 2, unlocking maximum performance for LLM and TTS inference workloads (e.g., Qwen3-TTS streaming, vLLM).

### Environment Specifications

The wheels in this repository are validated against the following software and hardware stack:
- **OS**: Linux (Ubuntu 24.04+)
- **GPU**: NVIDIA RTX 5090 (Compute Capability `sm_120`)
- **CUDA**: 13.0
- **PyTorch**: `2.12.0.dev*+cu130` (nightly)
- **Python**: 3.12

### ⚡ Installation

Install directly via `pip`:

```bash
# 1. Verify PyTorch 2.12 nightly + cu130 environment:
python -c "import torch; print(torch.__version__)"

# 2. Install the pre-built wheel from GitHub Releases:
pip install https://github.com/Skywarder0409/flash-attention-cu130-pt2.12-wheels/releases/download/v2.8.3/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl
```
*(Note: Replace the URL with the actual release asset link depending on the specific wheel version)*

---

<a id="chinese"></a>
## 🇨🇳 中文

适用于 **NVIDIA RTX 5090 (Blackwell 架构, 算力 12.0)** Linux 系统的 `flash-attn` 预编译 whl 安装包。

### 核心痛点

在全新的 Blackwell 架构（如 RTX 5090）和 CUDA 13.0 环境下，从源码级编译 `flash-attn` 常面临以下工程阻碍：
1. 本地编译耗时过长，通常超过 1 小时以上。
2. 极易因底层依赖缺失、系统 GCC 版本不兼容，或对 PyTorch nightly 测试版支持不足而导致构建失败。

本仓库提供预先编译完成的 `.whl` 文件，彻底绕过本地 C++ 扩展编译流程。实现秒级环境配置，确保业务场景（如 Qwen3-TTS 流式极速推理、vLLM 大模型部署等）快速落地并发挥极限性能。

### 运行时环境要求

本构建版本已在以下环境组合中经过充分验证：
- **操作系统**: Linux (Ubuntu 24.04+)
- **显卡**: NVIDIA RTX 5090 (算力 `sm_120`)
- **CUDA 版本**: 13.0
- **PyTorch 版本**: `2.12.0.dev*+cu130` (nightly)
- **Python 版本**: 3.12

### ⚡ 快速安装

通过 `pip` 直接安装 Release 附件：

```bash
# 1. 确认当前基础环境为 PyTorch 2.12 nightly + cu130:
python -c "import torch; print(torch.__version__)"

# 2. 直接从 GitHub Releases 远程拉取安装:
pip install https://github.com/Skywarder0409/flash-attention-cu130-pt2.12-wheels/releases/download/v2.8.3/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl
```
*(注意：请将具体下载链接替换为您实际上传的 Release Asset URL)*

---

### 💡 API Example / 代码示例 (Qwen3-TTS)

完成安装后，即可在各类深度学习框架中无缝激活推理加速。

```python
import torch
from qwen_tts import Qwen3TTSModel

# 1. Enable TF32 for Ampere+/Blackwell matrix multiplications (Up to 5x throughput)
#    激活 TF32 精度引擎 (矩阵乘法最高提升 5 倍吞吐)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Explicitly specify the attention implementation mechanism
#    初始化模型时显式指定注意力计算后端为 flash_attention_2
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```
