---
title: "GPU Utilization Is Not One Number: Tensor Cores, SM Active, and MFU"
description: "A practical guide to interpreting GPU-Util, SM Active, Tensor Core activity, MFU, and HFU without optimizing the wrong bottleneck."
date: 2026-07-30 15:28:00 -0700
series: GPU Performance
tags:
  - gpu
  - tensor-cores
  - performance
  - profiling
  - mfu
  - llm-infrastructure
---

Suppose an LLM job reports:

```text
GPU-Util:           99%
SM Active:          42%
Tensor Pipe Active: 21%
MFU:                18%
```

Is the GPU fully utilized, 42% utilized, 21% utilized, or 18% utilized?

All four numbers can be correct. To understand why, we first need the execution hierarchy that connects a model operation to the GPU hardware.

## First: threads, warps, blocks, grids, and SMs

When a framework launches a CUDA **kernel**, it does not send one indivisible task to the whole GPU. It launches a grid of thread blocks, and the GPU distributes those blocks across its Streaming Multiprocessors.

```text
Kernel launch
└── Grid
    ├── Thread block 0 ── assigned to one SM
    │   ├── Warp 0: threads 0–31
    │   ├── Warp 1: threads 32–63
    │   └── ...
    ├── Thread block 1 ── assigned to one SM
    └── ...

GPU
├── SM 0: resident blocks → warp schedulers → execution pipelines
├── SM 1: resident blocks → warp schedulers → execution pipelines
└── ...
```

The [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) defines this programming and execution model.

### Thread

A **thread** is one logical execution instance of a kernel. Threads have their own registers and indices, which let different threads process different elements of a tensor.

A thread is not independently scheduled like a CPU process. NVIDIA GPUs group threads into warps for instruction scheduling.

### Warp

A **warp** is a group of 32 CUDA threads. The SM's warp schedulers select eligible warps and issue their instructions to execution pipelines.

Threads in a warp follow the SIMT model: they execute the same instruction over different data lanes. If threads take different branch paths, the warp must execute the required paths with only the relevant lanes active, which can reduce efficiency.

This is why performance counters often discuss:

- **resident warps**, which currently occupy SM resources;
- **eligible warps**, which are ready to issue; and
- **issuing warps**, which actually issue an instruction.

### Thread block

A **thread block**, also called a Cooperative Thread Array or CTA, is a group of threads that can cooperate through shared memory and block-level synchronization.

The GPU assigns an entire block to one SM. The block remains on that SM for its lifetime; it is not split across SMs. An SM can host multiple blocks concurrently when registers, shared memory, warp slots, and architectural limits allow it.

Those resource limits determine how many blocks and warps can be resident, which is why block size, register use, and shared-memory use affect occupancy.

### Grid and kernel

A **kernel** is a GPU function executed in parallel by many threads. A kernel launch specifies the number and arrangement of blocks in the grid and the threads in each block.

A **grid** is the complete collection of blocks created by one kernel launch. The grid size determines how much parallel work is available to distribute across the GPU.

If a GPU has many SMs but a kernel launches only a few blocks, only a few SMs may receive work. The device can still show high GPU-Util because a kernel is continuously running, while SM Active remains low because much of the GPU has no assigned warp.

### Streaming Multiprocessor

A **Streaming Multiprocessor**, or **SM**, is the GPU's primary programmable execution unit. A simplified SM contains:

```text
Streaming Multiprocessor
├── Warp schedulers and dispatch units
├── Registers
├── Shared memory and L1 cache
├── CUDA-core arithmetic pipelines
├── Tensor Core pipelines
├── Load/store units
└── Other specialized pipelines
```

CUDA cores are arithmetic lanes inside an SM, not independently scheduled CPU-like cores. Tensor Cores are specialized matrix-math pipelines inside the SM. Warp instructions flow through the SM's schedulers to the appropriate pipelines.

An SM can therefore be **active** because it has an assigned warp without every arithmetic pipeline being busy. The warp may be waiting for memory, executing a non-Tensor instruction, stalled at a dependency, or using only part of the SM's available throughput.

### The mapping that matters for utilization

The relationship is:

```text
Kernel → grid → blocks → warps → threads
                   ↓
            blocks reside on SMs
                   ↓
          warp instructions use SM pipelines
```

This mapping explains several otherwise confusing observations:

- A small grid can produce high GPU-Util but low SM Active.
- Many resident warps can produce high occupancy but still stall.
- High SM Active does not prove Tensor Core activity.
- High Tensor-pipe activity does not prove high end-to-end model throughput.

## From the execution hierarchy to utilization metrics

Large matrix multiplications use Tensor Cores when the data type, layout, dimensions, and kernel support them. The same transformer block also runs softmax, normalization, activation, indexing, reduction, memory, and communication work that does not map entirely to Tensor Cores.

With the execution hierarchy established:

- **GPU-Util** asks whether any kernel was executing.
- **SM Active** asks whether an SM had at least one warp assigned.
- **Tensor-pipe activity** asks whether Tensor Core pipelines were active.
- **MFU** asks how much useful model computation finished relative to a declared hardware peak.

The word *utilization* hides the denominator. If the denominator changes, the percentage answers a different question.

<figure>
  <img src="{{ '/assets/images/gpu-utilization-metrics/metric-stack.svg' | relative_url }}" alt="Four metric layers showing GPU-Util, SM Active, Tensor pipe activity, and MFU with their different questions and scopes.">
  <figcaption>Figure 1. Original diagram by the author, based on NVIDIA NVML, DCGM, Nsight Compute, and published MFU definitions. The metrics are related but are not nested percentages.</figcaption>
</figure>

## Summary

1. **`GPU-Util` is a coarse device-busy signal.** NVIDIA defines it as time during a recent sample period when one or more kernels executed.
2. **SM Active is not SM throughput.** An SM counts active when it has at least one warp assigned, even if that warp is stalled.
3. **Occupancy is another metric.** It measures resident warps relative to capacity, not issued instructions or achieved throughput.
4. **Tensor-pipe activity is not Tensor Core FLOP efficiency.** “Any tensor pipe active” does not mean all Tensor Cores were fully loaded.
5. **MFU is calculated, not read from a hardware counter.** It divides useful model FLOPs completed per wall-clock second by an explicitly selected hardware peak.
6. **HFU can exceed MFU without improving training progress.** Activation recomputation performs additional hardware FLOPs that do not represent additional model work.
7. **The same time window matters.** A one-second NVML sample, a per-kernel Nsight report, and a five-minute job-level MFU cannot be compared as if they described one interval.
8. **Inference decode is often memory-bound.** Low Tensor Core activity or MFU can be expected even when token latency is excellent.

The metrics now map to distinct levels:

| Metric | Typical source | Core question |
|---|---|---|
| `GPU-Util` | NVML / `nvidia-smi` | Was at least one kernel executing? |
| SM Active | DCGM profiling | Did an SM have at least one warp assigned? |
| SM occupancy | DCGM / Nsight Compute | How many warps were resident relative to capacity? |
| Tensor-pipe active | DCGM / Nsight Compute | Was any Tensor pipeline active? |
| MFU | Application calculation | How much useful model math finished relative to peak? |

The first four are device/profiler measurements or counter-derived metrics; their exact collection mechanisms and denominators are tool- and version-specific. MFU is an accounting model built from throughput, FLOP assumptions, wall time, and a selected peak.

## GPU-Util: was any kernel running?

NVIDIA NVML defines GPU utilization as the percentage of a recent sample period during which **one or more kernels were executing**. The sample period is device-dependent and can range from roughly one second to one sixth of a second ([NVIDIA NVML: `nvmlUtilization_t`](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)).

The precise interpretation is:

> The device executed at least one kernel for approximately X% of NVML's recent sampling interval.

It does **not** mean:

- X% of SMs were used;
- X% of Tensor Cores were used;
- X% of peak FLOPs were delivered;
- X% of memory bandwidth was delivered; or
- X% of the work advanced the model.

### How 100% GPU-Util can still be slow

A continuous sequence of kernels can keep `GPU-Util` near 100% even when:

- each kernel has a tiny grid and reaches only a subset of SMs;
- kernels are dominated by memory latency or bandwidth;
- NCCL communication kernels occupy the device;
- matrix shapes poorly use Tensor Cores;
- the job performs recomputation or redundant work; or
- many short kernels create launch and synchronization overhead.

`GPU-Util` is excellent for detecting obvious idle periods. It is not a performance-efficiency score.

### The neighboring “memory utilization” metric is also coarse

NVML's memory utilization field reports the percentage of the sample period during which global device memory was being read or written. It is not the percentage of peak device-memory bandwidth delivered.

Use DCGM DRAM activity as a device-memory-interface busy-time signal, or use a profiler's memory-throughput metrics when the question is whether device-memory bandwidth is saturated.

## SM Active: did an SM have a warp assigned?

NVIDIA DCGM defines SM Active as the ratio of cycles in which an SM had **at least one assigned warp** ([NVIDIA DCGM field definitions](https://github.com/NVIDIA/DCGM/blob/72fa3feaa67d716a75323a8f47c34ff3ee73f824/dcgmlib/dcgm_fields.h)).

This is a presence metric at the SM-cycle level.

An SM can count active while its resident warp is:

- waiting for a device-memory load;
- waiting on a dependency;
- blocked at a barrier;
- executing address calculations;
- running CUDA-core instructions;
- running Tensor Core instructions; or
- participating in a communication kernel.

SM Active therefore answers a more detailed question than `GPU-Util`, but it still does not measure useful throughput.

### SM Active is not occupancy

**Occupancy** is the number of resident warps relative to the maximum resident-warp capacity. Registers, shared memory, thread-block size, and architecture limits determine how many warps can reside on an SM.

High occupancy can help hide latency by giving the scheduler other warps to run. It does not guarantee that those warps are eligible to issue instructions.

The [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) distinguishes:

- **resident/active warps** that occupy resources;
- **eligible warps** that are ready to issue; and
- **issuing warps** that actually issue instructions.

A high-occupancy kernel can be stalled. A fast Tensor Core kernel can use many registers, have lower occupancy, and still achieve excellent throughput.

### SM Active is not SM throughput

Nsight Compute throughput metrics estimate the use of execution pipelines relative to architecture-specific sustained peaks. Those metrics are closer to “how hard was the SM machinery driven?” than SM Active, but they still require the exact metric name and denominator.

Never translate “SM Active = 70%” into “70% of SM peak performance.”

## Tensor Core utilization: which metric do you mean?

“Tensor Core utilization” is not one universal counter.

DCGM exposes `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, defined as the ratio of cycles during which **any tensor pipe was active**. Nsight Compute exposes more detailed tensor-pipeline counters whose exact names and denominators depend on architecture and toolkit version.

Tensor-pipe activity is evidence that Tensor Core instructions ran. It is not the fraction of Tensor FLOP peak delivered.

### Why active cycles and Tensor FLOP/s differ

One active Tensor pipeline does not mean every Tensor Core pipeline is full. Achieved math rate also depends on:

- matrix dimensions and tile shapes;
- data type and accumulation mode;
- alignment and layout;
- padding and tail tiles;
- instruction scheduling;
- memory delivery;
- grid parallelism; and
- time spent outside Tensor Core kernels.

NVIDIA's [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains how arithmetic intensity, tile size, dimensions, and alignment affect GEMM performance.

### Do not subtract Tensor activity from SM Active

Suppose a dashboard reports:

```text
SM Active:          40%
Tensor Pipe Active: 21%
```

It is tempting to conclude that the remaining `19%` was CUDA-core work. That subtraction is not valid unless the tool explicitly guarantees identical scope, normalization, aggregation, and denominator.

Even when both metrics use elapsed cycles, “an SM had a warp assigned” and “any tensor pipe was active” are overlapping observations, not a partition of time into mutually exclusive categories.

Always retain:

- the full metric name;
- the collection tool and version;
- whether the value is per-kernel, per-range, per-device, or averaged;
- the measurement interval; and
- the denominator shown by the profiler.

## MFU: useful model FLOPs divided by selected peak

**Model FLOPs Utilization** is an application-level calculation:

<pre class="math-block" aria-label="Model FLOPs Utilization formula">
                 useful model FLOPs completed
MFU = ------------------------------------------------------
      wall time × sum of selected peak FLOP/s across GPUs
</pre>

For measurement duration `T` and GPUs with selected peaks `Pᵢ`:

<pre class="math-block" aria-label="Multi-GPU MFU formula">
MFU = F_model,useful / (T × Σ Pᵢ)
</pre>

The numerator counts algorithmic work completed once at the global-model level. The denominator uses the aggregate peak of the participating hardware.

The [PaLM paper](https://arxiv.org/abs/2204.02311) is a canonical large-scale reference for model and hardware FLOPs utilization terminology.

### A common dense-training approximation

For a dense decoder-only Transformer, a common shorthand is:

<pre class="math-block" aria-label="Approximate dense Transformer training FLOPs">
useful training FLOPs ≈ 6 × parameters × processed tokens
</pre>

The `6PT` estimate approximates forward and backward model math. It is not exact. Attention, sequence length, embeddings, vocabulary projection, normalization, routing, optimizer work, and architecture details can matter.

Use the same FLOP convention when comparing runs. A model-specific FLOP counter is preferable when comparing different architectures.

### A simple MFU example

Assume:

- eight GPUs;
- a selected peak of `1 PFLOP/s` per GPU;
- a two-second end-to-end interval; and
- `8 PFLOP` of useful model work completed.

The available peak work is:

```text
2 seconds × 8 GPUs × 1 PFLOP/s = 16 PFLOP
```

Therefore:

```text
MFU = 8 / 16 = 50%
```

This example is deliberately hardware-neutral. A real report must declare precision, SKU, dense or sparse mode, and clock policy.

## MFU versus HFU

**Hardware FLOPs Utilization** uses estimated or observed hardware-executed FLOPs rather than only useful model FLOPs:

<pre class="math-block" aria-label="Hardware FLOPs Utilization formula">
                 hardware-executed FLOPs
HFU = ------------------------------------------------------
      wall time × sum of selected peak FLOP/s across GPUs
</pre>

Activation checkpointing illustrates the difference. If the model requires `8 PFLOP` of useful work but recomputation makes the GPU execute `10 PFLOP` in the same two-second interval:

```text
MFU = 8 / 16  = 50%
HFU = 10 / 16 = 62.5%
```

HFU rose because the hardware did more work. The training algorithm did not make more progress.

Literature and implementations vary in how they estimate executed hardware FLOPs. Publish the exact inclusion rules instead of relying only on the label.

<figure>
  <img src="{{ '/assets/images/gpu-utilization-metrics/mfu-accounting.svg' | relative_url }}" alt="MFU accounting diagram separating useful model FLOPs from extra recomputation and comparing both with aggregate peak hardware work.">
  <figcaption>Figure 2. Original diagram by the author. MFU counts useful model work; HFU can additionally count executed recomputation or other charged hardware FLOPs.</figcaption>
</figure>

## Choosing the correct peak

An MFU number without its peak denominator is incomplete.

| Dominant execution mode | Candidate denominator | Common mistake |
|---|---|---|
| Dense BF16/FP16 Tensor Core | Exact SKU's dense BF16/FP16 Tensor Core peak | Using the higher sparse peak |
| TF32 Tensor Core | TF32 Tensor Core peak | Dividing by FP32 CUDA-core or FP16 peak |
| FP8 Tensor Core | Dense FP8 peak for operations actually executed in FP8 | Treating fallback BF16 layers as FP8 |
| INT8 inference | INT8 Tensor Core peak, with clearly defined operation counting | Comparing with BF16 training MFU |
| Structured 2:4 sparse | Sparse peak only when data and kernels use that mode | Assuming pruning automatically enables sparse peak |
| MIG instance | Peak corresponding to the allocated instance | Using the full physical GPU peak |

For monitoring, do not assume a parent-GPU `GPU-Util` can be apportioned to a MIG compute instance. NVIDIA documents limitations for GPU and memory utilization queries in `nvidia-smi dmon` on MIG-enabled GPUs; record the exact driver, command, GPU-instance, and compute-instance scope.

Also state whether peak means:

- **published/design peak**, which exposes clock and power loss; or
- **a separately reported clock-normalized peak**, computed from documented time-weighted effective SM clocks; it normalizes for delivered clocks and must not be compared directly with a published-peak MFU.

Record power limit, SM clocks, temperature, and throttle reasons beside the result. Do not silently switch denominators between experiments.

### Multi-GPU accounting

Use one global wall-clock interval:

<pre class="math-block" aria-label="Global multi-GPU MFU">
global useful model FLOPs
--------------------------------------------
wall time × sum of participating GPU peaks
</pre>

Do not average per-GPU MFUs without weighting. Do not multiply the global model FLOP count by tensor-parallel or pipeline-parallel degree; those GPUs split one model computation.

For data parallelism, count all globally processed tokens once.

### Dense versus MoE

For Mixture-of-Experts models, a total-parameter `6PT` calculation is usually misleading. Count the experts activated per token and disclose:

- active parameters or expert FLOPs;
- experts per token;
- capacity factor;
- padding or dropped tokens; and
- whether router FLOPs are included, plus the all-to-all communication time and bytes.

All-to-all is normally reflected in wall time, not added to the model-FLOP numerator. Expert GEMMs can be efficient while routing, communication, or load imbalance lowers end-to-end MFU.

## Why the metrics disagree

<figure>
  <img src="{{ '/assets/images/gpu-utilization-metrics/workload-scenarios.svg' | relative_url }}" alt="Four GPU workload scenarios comparing expected GPU-Util, SM Active, Tensor activity, memory pressure, and MFU.">
  <figcaption>Figure 3. Original diagram by the author. The patterns are qualitative; exact values depend on sampling, workload, architecture, and tool definitions.</figcaption>
</figure>

| Observation | Plausible explanation | Next measurement |
|---|---|---|
| GPU-Util high, SM Active low | Continuous small-grid kernels reach only part of the GPU | Kernel launch dimensions and Nsight Systems timeline |
| GPU-Util high, tensor activity low | Memory, reductions, optimizer, CUDA-core, or NCCL work dominates | Kernel-duration breakdown and DRAM throughput |
| SM Active high, SM throughput low | Warps are resident but stalled on memory, dependencies, or barriers | Eligible warps, stall reasons, roofline |
| Tensor activity high, MFU low | GEMMs are active, but communication, pipeline bubbles, data gaps, or recomputation dominate wall time | Phase-level timeline and HFU comparison |
| GPU-Util low, per-kernel tensor throughput high | Efficient GEMMs are separated by CPU, input, compilation, checkpoint, or synchronization gaps | Nsight Systems CPU/GPU timeline |
| MFU falls as GPUs are added | Smaller per-GPU GEMMs, collectives, imbalance, network contention, or pipeline bubbles | Per-rank timing and NCCL traces |

### Communication can look like GPU work

NCCL launches GPU kernels. Those kernels can raise `GPU-Util` and SM Active while Tensor activity and useful model progress remain low.

This is not “fake” activity—the GPU is doing communication work—but it is not model FLOP work. MFU captures the wall-clock cost because communication increases elapsed time without increasing the useful-model numerator.

## Training and inference need different expectations

### Training

Training MFU is useful because large forward and backward GEMMs often dominate useful work. End-to-end timing should include:

- forward and backward;
- communication;
- pipeline bubbles;
- optimizer phases;
- data stalls; and
- checkpoint or synchronization costs if they occur in the measured scope.

Excluding inconvenient phases can create an impressive number that does not represent training productivity.

### Prefill

LLM prefill processes many prompt tokens together. It often creates large GEMMs with enough parallelism and arithmetic intensity to use Tensor Cores well.

Prefill can therefore show:

- high GPU-Util;
- high SM Active;
- high tensor-pipe activity; and
- comparatively high math throughput.

### Decode

Autoregressive decode processes one new token per sequence at each step. At low batch or concurrency, operations become matrix-vector-like and repeatedly read model weights and KV cache.

Decode is often memory-bandwidth and latency bound. It can have:

- high GPU-Util;
- moderate or high SM Active;
- low Tensor-pipe activity;
- low training-style MFU; and
- excellent inter-token latency for the target workload.

The NVIDIA matrix guide explains why matrix-vector-like operations have low arithmetic intensity and are memory-limited.

For inference, report prefill and decode separately with:

- time to first token;
- inter-token latency or time per output token;
- request and token throughput;
- batch/concurrency distribution; and
- prompt and output length distributions.

Low decode MFU is not automatically a defect.

## A practical measurement workflow

### 1. Define one scope and interval

Choose:

- end-to-end training step;
- steady-state training window;
- prefill;
- decode;
- one kernel; or
- one NVTX-marked phase.

Do not compare a per-kernel tensor metric with a job-level MFU without accounting for the time outside that kernel.

### 2. Confirm ownership

Record:

- GPU UUID or MIG instance;
- processes and MPS clients;
- job/rank mapping; and
- whether another workload shares the device.

Device-level `GPU-Util` can include unrelated kernels.

### 3. Use GPU-Util as a liveness test

- **Low GPU-Util:** inspect CPU, data-loader, compilation, checkpoint, and synchronization gaps.
- **High GPU-Util:** continue investigating; the device is busy, not necessarily efficient.

### 4. Inspect the timeline

Use Nsight Systems or equivalent phase tracing to separate:

- GEMMs;
- attention and reductions;
- memory operations;
- NCCL;
- CPU gaps;
- synchronization; and
- prefill versus decode.

### 5. Inspect representative kernels

Use Nsight Compute for:

- launch/grid size;
- SM Active and throughput;
- occupancy;
- eligible and issuing warps;
- Tensor-pipe activity;
- memory throughput;
- arithmetic intensity; and
- stall reasons.

Profiling can perturb distributed workloads. Measure representative ranges and document profiler settings.

### 6. Calculate MFU independently

Freeze:

- FLOP model;
- global token/example count;
- wall-clock interval;
- GPU count;
- precision and execution mode;
- dense/sparse assumption; and
- peak source.

Recompute the number from raw inputs. MFU is not a dashboard counter to accept without its equation.

## Common misconceptions

- **“100% GPU-Util means peak performance.”** It means at least one kernel executed throughout the sampled interval.
- **“SM Active equals SM throughput.”** A warp can be assigned and stalled.
- **“High occupancy means a fast kernel.”** Residency is not issue rate or arithmetic intensity.
- **“Tensor-pipe active equals Tensor Core TFLOP/s.”** Active cycles do not reveal work delivered per cycle.
- **“SM Active minus Tensor activity equals CUDA-core time.”** The metrics are not guaranteed to partition one denominator.
- **“MFU comes from `nvidia-smi`.”** It is an application-level calculation.
- **“Use the largest number on the GPU spec sheet.”** The peak must match precision, mode, SKU, sparsity, and allocation.
- **“HFU above MFU means a better implementation.”** Recomputation can raise HFU without increasing useful progress.
- **“One MFU describes inference.”** Prefill and decode have different arithmetic intensity and objectives.
- **“Average GPU utilization is cluster MFU.”** Device-busy time is not useful global FLOP throughput.

## Decision checklist

Before calling a GPU “underutilized,” answer:

1. Which exact metric and tool produced the number?
2. What is its numerator and denominator?
3. What time window and aggregation were used?
4. Is the value device-wide, per-SM, per-kernel, or job-level?
5. Was the device shared by another process or communication workload?
6. Are SMs active, occupied, eligible, issuing, or near throughput peak?
7. Does “Tensor Core utilization” mean pipe-active cycles or achieved Tensor FLOP/s?
8. Is the workload compute-bound, memory-bound, launch-bound, or communication-bound?
9. Does MFU use useful model FLOPs or estimated executed hardware FLOPs?
10. Does the peak match the actual precision, sparse mode, SKU, MIG slice, and clock policy?
11. Are global tokens and model FLOPs counted exactly once across parallel ranks?
12. Are prefill and decode reported separately?

## Final mental model

> GPU performance metrics are layers of evidence, not competing answers to one question. GPU-Util tells you whether the device was idle, SM Active tells you whether SMs had assigned work, Tensor-pipe activity tells you whether Tensor Core pipelines participated, and MFU tells you how much useful model work finished relative to a declared peak.

Move through the layers in order:

```text
Is the GPU idle?
    ↓
Are the SMs receiving enough work?
    ↓
Are warps issuing or stalling?
    ↓
Which execution pipelines and memory paths are busy?
    ↓
Is that hardware activity producing useful end-to-end model progress?
```

No single percentage answers every step. Compare metrics over the same interval, preserve each metric's exact denominator, and judge the final result using the workload's real objective: training throughput, time to first token, inter-token latency, or request throughput.

## Quick knowledge check

Open each question to reveal the answer.

<details class="knowledge-check">
  <summary>1. Why can GPU-Util be 100% while the GPU is delivering far below peak performance?</summary>
  <div class="knowledge-check__answer">
    <p>GPU-Util measures whether one or more kernels executed during the sampled interval. A stream of small, memory-stalled, communication-heavy, or poorly shaped kernels can keep the device continuously busy without using all SMs, Tensor Core pipelines, or available FLOP capacity. Therefore, 100% GPU-Util proves that the GPU was not idle; it does not prove that the work was efficient.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>2. What is the key difference between GPU-Util and SM Active?</summary>
  <div class="knowledge-check__answer">
    <p>GPU-Util is a device-level sampled signal that asks whether any kernel was executing. SM Active is a finer-grained ratio that asks whether an SM had at least one assigned warp during a cycle. A small kernel can keep GPU-Util high while reaching only a fraction of the SMs, which produces much lower SM Active.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>3. Why are SM Active and occupancy not the same metric?</summary>
  <div class="knowledge-check__answer">
    <p>SM Active records whether an SM had at least one assigned warp. Occupancy measures how many warps are resident relative to the SM's maximum resident-warp capacity. Occupancy describes resource residency, while SM Active describes the presence of assigned work over time. Neither metric proves that warps were eligible to issue instructions or that the kernel achieved high throughput.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>4. How can SM Active be high while useful throughput remains low?</summary>
  <div class="knowledge-check__answer">
    <p>An SM counts as active even when its assigned warps are stalled on device-memory loads, dependencies, barriers, or communication. The GPU may therefore maintain many active SM cycles while issuing few useful instructions. Eligible-warps, issue-rate, stall-reason, memory-throughput, and roofline measurements are needed to explain the missing throughput.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>5. Why does Tensor-pipe activity not equal achieved Tensor Core TFLOP/s?</summary>
  <div class="knowledge-check__answer">
    <p>A Tensor-pipe activity metric usually reports cycles in which at least one Tensor pipeline was active. It does not say that every Tensor pipeline was full or that each active cycle delivered the maximum number of operations. Matrix dimensions, tile tails, precision, alignment, scheduling, memory delivery, and time spent outside GEMMs all affect achieved Tensor FLOP/s.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>6. Can Tensor-pipe activity be subtracted from SM Active to estimate CUDA-core work?</summary>
  <div class="knowledge-check__answer">
    <p>No. The two metrics represent overlapping observations and may use different scope, normalization, aggregation, and denominators. “An SM had a warp assigned” and “a Tensor pipeline was active” do not divide elapsed time into mutually exclusive categories. Use instruction- or pipeline-specific profiler metrics when you need an execution-unit breakdown.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>7. Why is MFU not available directly from <code>nvidia-smi</code>?</summary>
  <div class="knowledge-check__answer">
    <p>MFU is an application-level calculation, not a device-busy counter. It requires a useful-model FLOP estimate, a global amount of completed work, wall-clock time, participating GPU count, and a precision-appropriate hardware peak. Hardware telemetry alone cannot determine which executed operations represented useful model progress.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>8. Why can HFU be higher than MFU without improving training throughput?</summary>
  <div class="knowledge-check__answer">
    <p>MFU counts useful model FLOPs, while HFU can count additional hardware-executed FLOPs. Activation checkpointing, for example, recomputes forward operations during backward. That recomputation keeps the hardware busier and raises HFU, but it does not process more training tokens or complete more useful model work, so MFU does not increase by the same amount.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>9. Why must an MFU report declare its peak-FLOP denominator?</summary>
  <div class="knowledge-check__answer">
    <p>GPU peak rates differ by SKU, precision, Tensor Core mode, dense versus structured-sparse execution, MIG allocation, and clock assumption. Dividing dense BF16 work by an FP8 or sparse marketing peak makes the reported MFU artificially low and incomparable with correctly normalized results. The denominator must match the operations the workload actually executes.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>10. How should MFU be calculated for a distributed training job?</summary>
  <div class="knowledge-check__answer">
    <p>Count global useful model work once, then divide by one common wall-clock interval multiplied by the sum of the participating GPUs' selected peaks. Do not multiply model FLOPs by tensor- or pipeline-parallel degree because those ranks split one model computation. For data parallelism, count all globally processed tokens once rather than once per replica and then again globally.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>11. Why can low Tensor activity and low MFU be healthy during autoregressive decode?</summary>
  <div class="knowledge-check__answer">
    <p>Low-batch decode performs relatively little math per generated token while repeatedly reading model weights and KV-cache data. It is often constrained by device-memory bandwidth and latency rather than Tensor Core peak. A decode workload can therefore have low Tensor activity and training-style MFU while still meeting its intended inter-token latency and request-throughput targets.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>12. What does high Tensor activity combined with low MFU usually suggest?</summary>
  <div class="knowledge-check__answer">
    <p>It suggests that Tensor Core kernels are active during part of the run, but useful model math occupies too little of the full wall-clock interval. Communication, pipeline bubbles, data stalls, synchronization, small non-GEMM kernels, or activation recomputation may dominate the remaining time. Inspect a phase-level timeline and compare MFU with HFU before changing the GEMM kernels.</p>
  </div>
</details>

## Sources, attribution, and diagrams

This article uses the shared discussion as an editorial starting point, then verifies metric definitions against NVIDIA documentation and published MFU literature. All prose and diagrams are original; no source figure, table, or documentation passage is reproduced.

Metric names and availability vary across GPU architectures, drivers, DCGM releases, and Nsight Compute versions. Always retain the exact tool version and metric description with recorded values.

## References

1. **NVIDIA.** [*CUDA C++ Programming Guide*](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
2. **NVIDIA.** [*NVML API Reference*](https://docs.nvidia.com/deploy/nvml-api/), especially `nvmlUtilization_t`.
3. **NVIDIA.** [*NVIDIA System Management Interface (`nvidia-smi`) Documentation*](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
4. **NVIDIA.** [*Data Center GPU Manager field definitions*](https://github.com/NVIDIA/DCGM/blob/72fa3feaa67d716a75323a8f47c34ff3ee73f824/dcgmlib/dcgm_fields.h), profiling fields 1001–1005.
5. **NVIDIA.** [*Nsight Compute Profiling Guide*](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
6. **NVIDIA.** [*Matrix Multiplication Background User's Guide*](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
7. **Chowdhery, A.; et al.** [*PaLM: Scaling Language Modeling with Pathways*](https://arxiv.org/abs/2204.02311). 2022.
8. **Narayanan, D.; et al.** [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473). 2021.
9. **NVIDIA.** [*Transformer Engine FP8 Primer*](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

All links were accessed on July 30, 2026.

## Changelog

- **2026-07-30:** Moved the CUDA execution hierarchy before the utilization metrics and added definitions for threads, warps, blocks, grids, kernels, and SMs.
- **2026-07-30:** Added a final mental model and 12 expandable knowledge checks.
- **2026-07-30:** Initial publication.
