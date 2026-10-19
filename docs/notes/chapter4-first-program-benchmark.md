---
title: 第 4 章 · 第一个程序与性能分析
description: hello-gpu 第 4 章要点：完整 HIP 程序六步、线程与块的计算、event 计时、预热与重复、12N 有效带宽与算术强度、计时口径边界
---

# 第 4 章 · 第一个程序 + 性能分析

> Datawhale 85 期 hello-gpu · Part 0
> 对应教程：`docs/part0-intro/chapter4`

## 一句话

把向量加法写成完整 HIP 程序，再用 PyTorch 学「怎么测」：**先明确测的是哪个实现、区间从哪里到哪里，再谈数字。**

原文特别提醒：`vector_add.hip` 和 `benchmark_vector_add.py` 完成同样的数学运算，但 **Python 脚本没有调用那个 HIP 文件**。它们分别回答「怎样写」和「怎样测」，不能把后者的时间当成前者的性能。

## 完整 HIP 程序六步

```text
Host 准备 vector（1.0f / 2.0f）
  → hipMalloc 三块显存
  → hipMemcpy H2D（两个输入）
  → 启动 kernel
  → hipGetLastError + hipDeviceSynchronize
  → hipMemcpy D2H + 检查 + hipFree
```

```cpp
const int n = 1 << 20;                    // 1048576
const size_t bytes = n * sizeof(float);   // 一个数组的字节数

HIP_CHECK(hipMalloc(&d_a, bytes));
HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));

const int threads = 256;
const int blocks = (n + threads - 1) / threads;   // 向上取整
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
HIP_CHECK(hipGetLastError());
HIP_CHECK(hipDeviceSynchronize());
```

- `hipMalloc` 只分配空间，复制由 `hipMemcpy` 完成，是两件事。
- `<<<blocks, threads>>>` 指定启动规模，后面圆括号才是传给 kernel 的参数。
- 启动返回时 GPU 可能还没算完；`hipDeviceSynchronize()` 之后才能拷回结果。
- 想覆盖全部元素又允许尾部，分子加 `threads - 1` 再整除。

历史运行输出（2026-09-11，RX 9070 XT / 原生 Ubuntu 24.04.4）：

```text
device_name: AMD Radeon RX 9070 XT
vector_size: 1048576
blocks: 4096
threads_per_block: 256
max_error: 0
status: PASS
```

到这里只验证了结果正确，**还没有测这个手写 kernel 的时间**。

## PyTorch 计时：两种时钟

| | CPU 路径 | GPU 路径 |
| ---- | ---- | ---- |
| 计时 | `time.perf_counter()` 夹住 `torch.add` | `Event.record()` 夹住 `torch.add` |
| 读到的是什么 | 加法调用前后的 CPU 时钟 | 设备侧 event 间隔 |
| 注意 | 默认 `torch.set_num_threads(1)`，是单线程成绩 | 不含 H2D/D2H，也不是端到端耗时 |

```python
start.record()
torch.add(a, b, out=c)
end.record()
end.synchronize()
times.append(start.elapsed_time(end))
```

- `record()` 把标记放进当前执行流；`end.synchronize()` 等结束标记完成；之后读时间差。
- 同步之后做的是**读取时间差**，不是这时才开始计时。
- Event 会延迟初始化：原文在预热前先 `record()` 一次并同步，避免首次初始化进入测量。
- 预热 5 次、重复 30 次；报告 median（主比较值）、mean（看波动）、min（补充，不当作无干扰的真值）。
- 验证放在计时外，避免求和与比较混进加法时间。

## 实测结果（16,777,216 × float32）

| 项目 | CPU：i5-12600K，算子线程数 1 | GPU：RX 9070 XT |
| ---- | ---- | ---- |
| 中位数 | 8.24 ms | 0.340 ms |
| 平均值 | 8.27 ms | 0.349 ms |
| 最小值 | 8.08 ms | 0.339 ms |
| 有效带宽（median） | 24.4 GB/s | 592 GB/s |
| 有效带宽（min） | 24.9 GB/s | 594 GB/s |

## 12N 与算术强度

每个元素读 `a[i]`、读 `b[i]`、写 `c[i]`，各 4 字节：

$$
B_{\text{effective}} = N \times (4 + 4 + 4) = 12N \ \text{Byte}
$$

$$
BW_{\text{effective}} = \frac{12N}{t}, \qquad
AI = \frac{1\ \text{FLOP}}{12\ \text{Byte}} \approx 0.0833\ \text{FLOP/Byte}
$$

16,777,216 个元素 → 192 MiB = 201,326,592 字节；除以 0.339833 ms 得 592 GB/s（GB 按 10⁹ 字节）。

AMD 标称 RX 9070 XT 最高 640 GB/s，592 / 640 ≈ 92.6%。

## 口径边界（本章最重要的一段）

- `12N` 是**算法要求的有效字节数**，不是硬件计数器测出的显存流量。缓存、访存粒度、写入路径都可能让物理流量与它不同。
- 92.6% 是**模型下的对照**，不能称为显存控制器的实测利用率，也不能据此判断「只剩 7.4% 优化空间」——本章没有采集显存流量计数器。
- CPU 时间包含 Python 到算子实现的调用开销；GPU 是设备侧 event 间隔。两者不能直接当作「整个应用换到 GPU 的加速比」。
- GPU event 区间不含输入输出拷贝，也不代表整个 Python 程序的耗时。短运算下提交节奏和设备状态也会影响间隔。
- 小向量时，启动、提交和缓存也可能成为主要因素，**不能把「算术强度低」等同于「任意规模都只受带宽限制」**。
- launch / dispatch 开销本章没有单独测量，原文也没有给出具体微秒数。

## 规模效应（同一轮实测，GPU 路径）

| N | 每个数组 | GPU 中位数 | 有效带宽 |
| ---- | ---- | ---- | ---- |
| 1,048,576 | 4 MiB | 0.0241 ms | 522 GB/s |
| 16,777,216 | 64 MiB | 0.340 ms | 592 GB/s |
| 67,108,864 | 256 MiB | 1.32 ms | 608 GB/s |

数组变大后总时间增加，有效带宽也在变化；小规模时平均值与最小值差距更大，一个数不足以概括运行情况。

## 复跑记录清单

日期、CPU/GPU 型号、系统与 ROCm/PyTorch 版本、源码版本、输入规模、线程配置、warmup/repeat、正确性结果、三种时间统计量；带宽旁注明用 median 还是 min、是否包含数据搬运。这样过几天仍能判断两组结果是否可比较。

## 本章小结

- 完整程序包含主机准备、设备存储、数据拷贝、kernel 启动、等待和结果检查；kernel 只是其中一部分。
- 全局下标决定访问位置，`idx < n` 保护最后一个 block 的无效线程。
- GPU event 标记区间，同步保证区间完成；预热后重复测量，用中位数配合平均值和最小值阅读。
- `12N / t` 给出有效带宽，不能代替物理显存流量或端到端程序耗时。
- 下一章进入 benchmark 与可信计时（第 5 章）。

## 原文出处

- 教程正文：<https://datawhalechina.github.io/hello-gpu/part0-intro/chapter4/>
- 配套代码：`code/part0-intro/chapter4/`（`vector_add.hip`、`benchmark_vector_add.py`）
