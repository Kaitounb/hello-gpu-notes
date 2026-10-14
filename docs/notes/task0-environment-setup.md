---
title: Task 0 · 环境准备（第 1 章）
description: hello-gpu 第 1 章要点：三道环境验证门、实验基线、关键命令、最小 HIP 骨架与排错顺序
---

# Task 0 · 环境准备（第 1 章）

> Datawhale 85 期 hello-gpu · Task 0
> 对应教程：第 1 章 环境准备（`docs/part0-intro/chapter1`）

## 一句话

本章是一张「开工前体检表」：**不要求你懂 ROCm / HIP / uv 的原理**，只验证三件事——

1. ROCm 能不能看到 GPU？
2. PyTorch 能不能用上 GPU？
3. 最小 HIP 程序能不能编译运行？

三道门依次推开，基础链路就算打通了。

## 三道门

| 门 | 验证什么 | 通过后说明 |
| ---- | ---- | ---- |
| 第一道门 | ROCm 能看到 GPU | 底层驱动和运行时基本可用 |
| 第二道门 | PyTorch ROCm | 框架层能把计算放到 AMD GPU 上 |
| 第三道门 | 最小 HIP 程序 | 后续手写 kernel 的路径基本打通 |

每一道都是上一道的延伸，**跳不过去**。

## 实验基线

| 项目 | 基线 |
| ---- | ---- |
| 硬件 | AMD Radeon RX 9070 XT |
| GPU 架构 | gfx1201（RDNA4；ISA 名 `gfx12-generic`） |
| ROCm SDK | 10.0.0 |
| HIP 组件版本 | 7.15.26333 |
| 操作系统 | 原生 Ubuntu 24.04（x86_64） |
| Python 环境管理 | uv 0.11.28 |
| ROCm Python 包来源 | AMD `https://stable.repo.amd.com/rocm/whl-next/` |

注意 **ROCm SDK 和 HIP 组件各有版本号**：装 ROCm 10.0 后看到 HIP `7.15.26333` 是正常输出。本章验证于 **2026-09-19** 复测（版本对不上不影响验证顺序，只是设备名和工具输出会有差异）。

## 关键命令

```bash
# 同步本篇环境（一条命令，不用手动装 torch）
cd code/part0-intro
uv sync
source ./activate-rocm.sh

# 第一道门：ROCm 能看到 GPU 吗
rocminfo | grep -E "^[[:space:]]*(Name|Marketing Name|Vendor Name|Device Type|Compute Unit):" | head -20

# 第二道门：PyTorch 能用 GPU 吗
python chapter1/check_torch_rocm.py

# 第三道门：最小 HIP 程序
hipcc --version
cd chapter1 && hipcc vector_add.hip -O2 -o vector_add && echo "compile_status: PASS"
./vector_add
```

## 三道门分别看什么

**第一道门 —— `rocminfo`**：关键看两点，`Device Type: GPU` 必须出现，且 `Name` 是 `gfx1201`（对应 Marketing Name `AMD Radeon RX 9070 XT`）。两条对上，说明驱动认得你的卡。

**第二道门 —— `check_torch_rocm.py`**：脚本只做三件事——导入 PyTorch、检查 GPU 后端、在 GPU 上跑一次 `1024 × 1024` 矩阵乘。看到 `cuda_available: True` 和 `device_name: AMD Radeon RX 9070 XT` 就过了。

- 这里有个新手必问点：**为什么 ROCm 版 PyTorch 里到处是 `cuda`？** 答案是历史包袱——PyTorch 设备字符串一直沿用 `cuda`，没为 ROCm 单开一条。`device="cuda"` 不代表你在用 NVIDIA 卡，读作「把 tensor 放到当前可用的 GPU 后端」即可。

**第三道门 —— `vector_add.hip`**：`hipcc` 编译通过意味着编译器认识 gfx1201、能找到 device library；运行后看到 `status: PASS` 且 `max_error: 0`，说明 kernel 启动、H2D/D2H 拷贝、结果校验全部正常。

## 最小 HIP 程序骨架（记在心里）

几乎所有 HIP / CUDA 程序都是这个壳，变的只是 kernel 内部那几行：

```text
Host 准备输入
  → Device 分配显存
  → Host 到 Device（H2D）拷贝
  → 启动 HIP Kernel
  → Device 到 Host（D2H）拷贝
  → 检查结果
  → 释放资源
```

几个概念（第 2 篇才细讲，这里只要认个脸）：

- **Host（主机）**：CPU + 系统内存，负责准备输入、发起 GPU 任务、检查结果。
- **Device（设备）**：GPU + 显存；HIP Runtime 和驱动是两者之间的软件桥梁。
- **HIP Kernel**：在 GPU 上并行执行的计算函数，**只负责计算，不含前后的数据拷贝**。
- 启动后 Host 不一定会原地等，示例用 `hipDeviceSynchronize()` 等 GPU 算完再拷回结果。

## 平台边界

- 本章验证是**原生 Ubuntu 24.04**；**WSL2 需要按 AMD 安装页单独验证，原生结果不能直接当 WSL2 的结果。**
- 「计算能跑」不等于「状态监控也能跑」：`rocminfo`、PyTorch、HIP 通过只说明计算路径可用；`rocm-smi`、`amd-smi` 和硬件性能计数器依赖平台驱动接口，要另测。本章必做项只有 `rocminfo` + PyTorch + 最小 HIP。

## 排错铁律

**先确认底层，再确认上层**，顺序别反：

```text
ROCm / GPU 可见性
  → Python 环境
  → PyTorch ROCm
  → HIP / Triton / profiling 工具
```

(`rocminfo` 都看不到 GPU 时，先别去跑 PyTorch/HIP/Triton —— 底层不通，上层报错只会更迷惑。)

求助时别只甩一句「跑不通」，至少带上：**机器信息 / 目录 / 是否 source 了环境 / 实际命令 / 期望结果 / 完整报错 / 最近改动**。用 `tee` 存日志：

```bash
python chapter1/check_torch_rocm.py 2>&1 | tee check_torch_rocm.log
```

顺手记下：ROCm 版本、Python 版本、PyTorch 版本、uv 环境路径、GPU 架构（gfx1201）、运行日期。

## 本章小结

- 三道门：**ROCm 可见 → PyTorch ROCm → 最小 HIP 路径**，层层递进。
- 当前基线是原生 Ubuntu 24.04 + ROCm 10.0，其他平台单独验证。
- 环境由 `pyproject.toml` + `uv.lock` 固化，进目录 `uv sync` 即可复现。
- `activate-rocm.sh` 的核心职责是让 `ROCM_PATH` 指向 `_rocm_sdk_devel`（不是 `_rocm_sdk_core`）。
- `cuda:0` 是历史命名，不代表 NVIDIA 卡。
- 排错从下往上，求助带全信息。
- 下一章进入 GPU 体系结构：CU、wavefront、LDS。

## 原文出处

- 教程正文：<https://datawhalechina.github.io/hello-gpu/part0-intro/chapter1/>
- 配套代码：`code/part0-intro/chapter1/`（`check_torch_rocm.py`、`vector_add.hip`）
