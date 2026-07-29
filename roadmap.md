---
title: Editorial Roadmap
permalink: /roadmap/
eyebrow: 26-week learning path
description: A dependency-aware sequence from hardware fundamentals to reliable, production-scale LLM systems.
---

The roadmap is designed as a cumulative course: later essays reuse the performance models, terminology, and experiments established earlier. The default cadence is one deeply researched article per week, but accuracy and reproducibility take priority over a fixed date.

## Editorial standard

Every major article should answer four questions:

1. **How does it work?** Build the mechanism from first principles.
2. **What is the bottleneck?** Quantify compute, memory, communication, latency, or cost.
3. **When should I use it?** Compare realistic alternatives and failure modes.
4. **How can I verify it?** Include commands, code, traces, or benchmark methodology.

Articles will use original prose and diagrams, cite primary sources, distinguish measured results from estimates, and record hardware, software versions, and benchmark settings.

## Compute Foundations
{: #compute-foundations }

### Weeks 1–4: Build the machine-level mental model

1. **[Linux, Networking, and Storage for AI Engineers]({{ '/posts/2026-07-29-linux-networking-storage-ai-engineers/' | relative_url }})** — Published July 29, 2026<br>
   Processes, virtual memory, page cache, NUMA, filesystems, sockets, TCP, DNS, block storage, object storage, and the diagnostic tools that connect them.<br>
   *Build artifact:* a repeatable host and I/O diagnostic checklist.

2. **GPU Architecture from First Principles**<br>
   Streaming multiprocessors, warps, schedulers, tensor cores, occupancy, latency hiding, and the roofline model.<br>
   *Build artifact:* a worksheet that predicts whether a kernel is compute- or bandwidth-bound.

3. **VRAM Fundamentals and the CUDA Memory Hierarchy**<br>
   Registers, local memory, shared memory, L1/L2 cache, HBM, coalescing, pinned host memory, unified memory, and allocation behavior.<br>
   *Build artifact:* CUDA memory microbenchmarks and a model-memory calculator.

4. **Quantization without Hand-Waving**<br>
   FP32, TF32, BF16, FP16, FP8, INT8, INT4, scaling, calibration, weight-only quantization, activation quantization, kernels, and accuracy trade-offs.<br>
   *Build artifact:* a precision-selection matrix for training and inference.

## Fast LLM Inference
{: #fast-llm-inference }

### Weeks 5–12: Explain where every millisecond and byte goes

5. **The Lifecycle of an LLM Token**<br>
   Tokenization, prefill, decode, attention, sampling, time to first token, inter-token latency, tokens per second, and arithmetic intensity.<br>
   *Build artifact:* a latency and throughput measurement harness.

6. **Batching and Continuous Batching**<br>
   Static batching, dynamic batching, iteration-level scheduling, head-of-line blocking, fairness, queueing, and latency-throughput trade-offs.<br>
   *Build artifact:* a batching simulator for mixed prompt and generation lengths.

7. **KV Caching, Paged Attention, and Prefix Caching**<br>
   KV memory math, fragmentation, block managers, cache eviction, shared prefixes, cache-aware routing, and long-context pressure.<br>
   *Build artifact:* a capacity calculator for concurrent requests.

8. **Speculative Decoding and Other Ways to Generate Faster**<br>
   Draft models, acceptance rates, tree-based speculation, prompt lookup decoding, Medusa-style heads, and when speculation loses.<br>
   *Build artifact:* a break-even model for draft-model selection.

9. **vLLM Internals and Performance Tuning**<br>
   PagedAttention, scheduling, tensor parallelism, chunked prefill, memory utilization, quantization, and operational tuning.<br>
   *Build artifact:* a reproducible vLLM benchmark suite.

10. **TensorRT-LLM, SGLang, and llama.cpp**<br>
    Compilation, fused kernels, structured generation, radix/prefix caching, CPU and edge inference, hardware targets, and engine selection.<br>
    *Build artifact:* an engine comparison matrix using the same model and workload.

11. **Multi-GPU and Multi-Node Inference**<br>
    Tensor, pipeline, data, sequence, and expert parallelism; communication costs; disaggregated prefill/decode; and replica placement.<br>
    *Build artifact:* a parallelism decision tree and topology-aware deployment plan.

12. **Benchmarking and Capacity Planning for LLM Serving**<br>
    Workload distributions, warmup, percentiles, concurrency, goodput, saturation, SLOs, GPU utilization, and cost per million tokens.<br>
    *Build artifact:* a benchmark report template that prevents misleading comparisons.

## Distributed AI
{: #distributed-ai }

### Weeks 13–16: Scale computation across devices and hosts

13. **NCCL, NVLink, PCIe, InfiniBand, and RDMA**<br>
    Rings, trees, collectives, topology discovery, GPUDirect, congestion, bandwidth-delay product, and common failure signatures.<br>
    *Build artifact:* a topology and collective-performance troubleshooting playbook.

14. **DDP, FSDP, DeepSpeed, and ZeRO Explained**<br>
    Gradient synchronization, parameter sharding, optimizer-state sharding, activation checkpointing, CPU/NVMe offload, and memory-speed trade-offs.<br>
    *Build artifact:* a training-memory estimator and strategy comparison.

15. **LoRA, QLoRA, PEFT, and Fine-Tuning Pipelines**<br>
    Low-rank updates, adapter placement, quantized base models, optimizer choices, data formatting, evaluation, merging, and serving adapters.<br>
    *Build artifact:* an end-to-end fine-tuning pipeline with experiment metadata.

16. **Docker, Kubernetes, and GPU Orchestration**<br>
    Images, layers, runtimes, NVIDIA Container Toolkit, device plugins, scheduling, taints, topology, MIG, health checks, autoscaling, and rollouts.<br>
    *Build artifact:* a production-ready GPU workload deployment checklist.

## Serving Platforms
{: #serving-platforms }

### Weeks 17–20: Operate models as dependable services

17. **Triton, vLLM, KServe, Ray Serve, and SGLang Serving**<br>
    Server responsibilities, model repositories, request routing, autoscaling, composition, streaming, rollout patterns, and platform boundaries.<br>
    *Build artifact:* a decision matrix and reference service architecture.

18. **Observability for LLM Systems**<br>
    OpenTelemetry traces, Prometheus metrics, Grafana dashboards, structured logs, token-level telemetry, Langfuse, Phoenix, sampling, and cardinality.<br>
    *Build artifact:* an observability schema spanning gateway, model server, GPU, and application.

19. **CI/CD for ML, MLflow, and Model Registries**<br>
    Data and model versioning, reproducible builds, evaluation gates, registry stages, provenance, canaries, rollback, and policy enforcement.<br>
    *Build artifact:* a model promotion pipeline with explicit release evidence.

20. **Kafka and Streaming Inference Pipelines**<br>
    Partitions, consumer groups, ordering, backpressure, micro-batching, retries, idempotency, dead-letter queues, and online feature joins.<br>
    *Build artifact:* a streaming inference design with failure and replay semantics.

## Production LLM Systems
{: #production-llm-systems }

### Weeks 21–26: Build useful, economical, and safe applications

21. **Embeddings, Vector Databases, and RAG Pipelines**<br>
    Chunking, embedding models, approximate nearest-neighbor indexes, hybrid retrieval, reranking, context construction, freshness, and evaluation.<br>
    *Build artifact:* a measurable RAG baseline with retrieval and answer-quality metrics.

22. **Prompt Caching, Semantic Caching, and Cost Optimization**<br>
    Exact and prefix caches, semantic similarity, invalidation, tenant isolation, cache economics, token budgets, and quality-risk controls.<br>
    *Build artifact:* a cache policy with break-even and correctness analysis.

23. **Model Routing and Fallback Strategies**<br>
    Static rules, quality and cost routers, cascades, confidence, hedging, circuit breakers, regional failover, and graceful degradation.<br>
    *Build artifact:* an SLO-aware routing policy tested against workload traces.

24. **LLM Evaluation, Benchmarking, and A/B Testing**<br>
    Golden sets, rubric-based judging, pairwise comparison, judge bias, offline-online gaps, statistical power, guardrail metrics, and experiment design.<br>
    *Build artifact:* an evaluation scorecard and production experiment template.

25. **MCP, AI Agents, and Workflow Orchestration**<br>
    Tools, resources, prompts, state, planning, memory, retries, durable execution, human approval, protocol boundaries, and agent observability.<br>
    *Build artifact:* a bounded agent workflow with explicit permissions and failure recovery.

26. **Security, Guardrails, and Prompt-Injection Mitigation**<br>
    Threat modeling, indirect prompt injection, data exfiltration, tool authorization, sandboxing, content controls, secrets, audit trails, and incident response.<br>
    *Build artifact:* a layered security architecture and red-team test plan.

## Publication workflow

1. Define the reader question and measurable learning outcome.
2. Collect primary sources: papers, official documentation, specifications, and source code.
3. Reproduce the mechanism with a small experiment before writing conclusions.
4. Create original diagrams and record every benchmark assumption.
5. Draft from the supplied post template, then verify links, commands, numbers, and claims.
6. Publish with a changelog and update the article when major tools or APIs change.
