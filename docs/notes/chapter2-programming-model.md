---
title: 第 2 章 · 编程模型与 wavefront 执行
description: hello-gpu 第 2 章要点：host/device 分工、kernel 与 thread、grid/block/wavefront/lane、索引公式、边界检查、逻辑线程与物理执行
---

# 第 2 章 · GPU 体系结构（上）：编程模型与 wavefront 执行

> Datawhale 85 期 hello-gpu · Part 0
> 对应教程：`docs/part0-intro/chapter2`

## 一句话

本章回答三个问题：**一次计算由谁发起？只写一份规则，为什么会有很多执行者？每个执行者怎么找到自己负责的数据？**

## 谁发起、谁执行

- **Host（主机端）**：CPU 一侧，准备输入、把数据放到 GPU 能访问的位置、发起计算、确认完成、取回结果。
- **Device（设备端）**：GPU 一侧，执行计算。
- **Kernel（核函数）**：写在 GPU 上运行的计算规则，只写一份。
- **Thread（线程）**：这份规则的一次执行，可以有很多份。

CPU 提交一次 GPU 计算的动作叫**启动（Launch）**。启动只负责提交工作；要使用结果，还得确认 GPU 已经算完。

## 组织关系

| 名字 | 含义 | 本章手算例子 |
| ---- | ---- | ---- |
| Grid | 本次启动的全部线程块 | 2 个 Block |
| Block | 一组线程，必要时可在块内协作 | 64 个线程 |
| Wavefront | 块内共同执行指令的小组 | wave32 → 2 个 |
| Thread | kernel 的一次执行 | 处理分配给自己的数据 |

记法：营 → 连 → 排 → 单兵。包含关系是 **Grid ⊃ Block ⊃ wavefront ⊃ Thread**；Kernel 是这些线程共同执行的程序，不是包含关系里再加的一层。

- **Lane**：线程在 wavefront 内的位置。
- 启动时只指定**块数**和**每块线程数**；wave 大小由 kernel 采用的配置决定，不需要单独指定。
- 不同 Block 的线程不会拼成同一个 wavefront。

## 线程编号与索引

HIP 让每个线程自己读取身份，不需要逐个填写：

| 代码里的名字 | 含义 | 例子（块 1、每块 4 线程、块内 2 号） |
| ---- | ---- | ---- |
| `blockIdx.x` | 我属于第几块 | 1 |
| `blockDim.x` | 每块有多少个线程 | 4 |
| `threadIdx.x` | 我在本块里是第几个 | 2 |

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 1×4+2 = 6
```

编号只是身份，**索引规则才决定数据位置**。编号也不表示执行先后：第 1 块不需要等第 0 块算完。

## 边界检查

长度 10、每块 4 线程 → 需要 3 块、12 个线程，多出 2 个：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

越界线程**仍然存在**，只是靠 `if (idx < n)` 跳过数组读写。所以启动的线程数和数组长度可以不同，但要保证有效位置都有人处理、多余线程安全跳过。

读 kernel 时始终站在**一个线程**的角度：先算出自己负责的位置，检查越界，再取两个数、相加、写回。

## 线程编号与 lane

块内线程 35，在 wave32 下属于第二个 wavefront：`35 − 32 = 3`，即 **lane 3**，不是 lane 35。

| 块内线程号 | wavefront 编号 | lane 编号 |
| ---- | ---- | ---- |
| 31 | 0 | 31 |
| 32 | 1 | 0 |
| 35 | 1 | 3 |
| 63 | 1 | 31 |

每块 256 线程、wave32 → 8 个 wavefront，每个都是 lane 0～31。每块只有 4 线程时，它仍只占一个 wavefront 的 4 个位置，不会把 wave 大小改成 4，也不会跨 Block 补满。

`gfx1201` 同时支持 wave32 和 wave64，具体要看 kernel 采用哪一种，**不能把 32 当作所有 AMD GPU 程序的固定值**。

## 分组与硬件

| 名字 | 含义 |
| ---- | ---- |
| SIMD | 用一条向量指令处理一组 lane 的数据 |
| CU | 包含执行单元及相关资源的硬件层 |
| WGP | RDNA 中包含 CU 的更外层组织，也与同一工作组的放置范围有关 |

LLVM AMDGPU 文档给出的约束：同一工作组的 wavefront 在同一 WGP 中执行；更细的执行范围还与 CU/WGP 模式有关，具体模式要看目标与编译产物，不能仅从 block 大小反推。

**逻辑上有多少线程，与物理上此刻能执行多少，是两个问题。** 把 block 调大不会凭空增加计算设备。

一个容易困惑的数据点：AMD 产品规格列 RX 9070 XT 有 64 个物理 CU，而 `hipDeviceProp_t::multiProcessorCount` 读到的是 32。两个字段统计口径不同，查硬件时要同时看字段含义和目标平台。

## 名称对照（有 CUDA 基础时）

thread / work-item、block / workgroup、warp / wavefront、grid、shared memory / LDS。名称能对应，不代表两个厂商的硬件数量、wave 大小或性能结果相同。

## 容易混

- wavefront 不是新增的线程，只是原有线程的执行分组。
- 每个线程执行一遍 kernel，不是执行 kernel 的某一步。
- 线程编号、lane 编号都不直接指定数组地址，数据位置仍由索引规则计算。

## 本章小结

- CPU 准备数据、启动 kernel，并在使用结果前确认 GPU 已完成。
- Kernel 是共同规则，Thread 是它的一次执行；启动规模 = 块数 × 每块线程数。
- Grid ⊃ Block ⊃ wavefront ⊃ Thread；Lane 是线程在 wavefront 内的位置。
- 索引公式把编号映射到数据，边界判断保护最后一块的越界线程。
- 逻辑线程数 ≠ 同时执行的线程数；Block 与 wavefront 是线程分组，CU/WGP/SIMD 是承接工作的硬件。

## 原文出处

- 教程正文：<https://datawhalechina.github.io/hello-gpu/part0-intro/chapter2/>
- 延伸：HIP 编程模型、LLVM AMDGPU Usage、ROCm GPU specifications（链接见原文「延伸阅读」）
