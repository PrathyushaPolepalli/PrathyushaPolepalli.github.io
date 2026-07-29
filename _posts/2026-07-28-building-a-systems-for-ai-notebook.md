---
title: Building a Systems-for-AI Notebook
description: Why this site starts at hardware constraints and works upward to reliable production LLM systems.
series: Site Notes
tags:
  - ai-systems
  - gpu
  - llm-inference
  - distributed-systems
---

The performance of an AI application is never determined by the model alone. It emerges from the interaction between hardware, kernels, memory, networking, runtimes, schedulers, serving infrastructure, and the workload itself.

This notebook is an attempt to make those interactions easier to reason about.

## The organizing idea

The articles will move upward through the stack:

1. **Hardware constraints** such as compute throughput, memory bandwidth, VRAM capacity, and interconnect topology.
2. **Runtime mechanisms** such as CUDA kernels, collectives, batching, caching, quantization, and parallelism.
3. **Serving systems** that schedule requests, place models, expose APIs, collect telemetry, and recover from failures.
4. **Application systems** that retrieve context, route models, evaluate quality, orchestrate tools, and enforce security boundaries.

This order matters. It is difficult to tune continuous batching without understanding prefill and decode. It is difficult to size a KV cache without understanding attention memory. It is difficult to choose tensor parallelism without understanding the communication path between GPUs.

## What “practical” means here

A useful systems explanation should let a reader predict behavior before running a benchmark. Each major note will therefore try to include:

- a compact mental model;
- explicit assumptions and back-of-the-envelope math;
- original architecture or data-flow diagrams;
- a reproducible experiment;
- failure modes and misleading metrics;
- a decision checklist; and
- references to primary sources.

Framework APIs will change. The underlying constraints change much more slowly. The goal is to explain both.

## Where the series starts

The first sequence covers Linux and I/O fundamentals, GPU architecture, CUDA memory, and quantization. From there, the roadmap moves into token generation, batching, KV caches, speculative decoding, inference engines, distributed execution, serving platforms, and production reliability.

The full dependency-aware sequence is available in the [editorial roadmap]({{ '/roadmap/' | relative_url }}).
