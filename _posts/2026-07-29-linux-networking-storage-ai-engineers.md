---
title: Linux, Networking, and Storage for AI Engineers
description: "A first-principles map of the host data path, from virtual memory and NUMA to TCP, filesystems, object stores, pinned memory, and GPU input stalls."
date: 2026-07-29
series: Compute Foundations
tags:
  - linux
  - networking
  - storage
  - gpu
  - performance
  - ai-infrastructure
---

A GPU can sit idle even when the host has spare CPU, the storage dashboard looks green, and the network link is nowhere near its advertised bandwidth. This is not contradictory. Each metric observes one layer of a pipeline, while the GPU waits for the slowest dependency of the next batch.

To debug that pipeline, an AI engineer needs a working model of the host beneath the framework:

- what a Linux process actually contains;
- why virtual memory is not physical memory;
- how the page cache changes storage behavior;
- why two devices in the same server can be far apart;
- where TCP queues and retransmissions hide;
- how filesystems, block devices, and object stores differ; and
- how bytes move from storage into GPU memory.

This article builds that model and ends with a read-only diagnostic workflow. It intentionally avoids changing kernel settings or presenting a universal tuning recipe. The correct optimization depends on the workload, topology, cache state, and service-level objective.

<figure>
  <img src="{{ '/assets/images/linux-networking-storage/end-to-end-data-path.svg' | relative_url }}" alt="Conceptual data path from remote or local storage through the Linux kernel, user-space data workers, pinned host memory, and GPU memory.">
  <figcaption>Figure 1. Original diagram by the author, based on Linux memory-management, networking, PyTorch data-loading, and NVIDIA CUDA/GDS concepts. Conceptual and not to scale.</figcaption>
</figure>

## Summary

1. **Treat the host as a pipeline and a topology, not a single resource.** Storage, network, CPU transforms, memory placement, host-to-device copies, and GPU kernels can each set the pace.
2. **`VmSize` is address space, not RAM.** Use RSS composition, cgroup limits, faults, reclaim, and swap activity together.
3. **Low free memory is not automatically bad.** Linux uses otherwise-idle RAM as page cache; `MemAvailable` is usually more informative than `MemFree`.
4. **Cache state is part of the experiment.** A warm second epoch and a cold first epoch are different workloads.
5. **Same host does not mean same distance.** NUMA nodes and PCIe root complexes determine the path between CPUs, RAM, GPUs, NICs, and NVMe devices.
6. **Bandwidth does not erase latency.** TCP needs enough bytes in flight to fill a high-bandwidth, high-RTT path, and retransmissions can destroy tail latency without saturating the link.
7. **Storage has three separate dimensions:** throughput, IOPS, and latency. Report request size, concurrency, cache state, and access pattern with them.
8. **Pinned memory and GPUDirect Storage are conditional tools.** Both can help, but neither makes an inefficient pipeline automatically fast.
9. **Diagnose with synchronized deltas.** Compare process, cgroup, memory, network, storage, topology, and GPU signals over the same reproducible interval.

## The end-to-end AI data path

Consider one training batch stored in an object store:

1. A client resolves an endpoint and establishes or reuses a network connection.
2. TCP delivers a byte stream into kernel socket buffers.
3. A storage client, filesystem, or application library turns those bytes into file or object reads.
4. Data lands in kernel page-cache pages or user-space buffers.
5. Data-loader workers parse, decompress, decode, tokenize, augment, and collate samples.
6. The runtime may copy the batch into page-locked, or **pinned**, host memory.
7. DMA transfers the batch across PCIe or another supported interconnect into GPU memory.
8. GPU kernels finally consume it.

Local NVMe removes the remote network hop, but not necessarily filesystem metadata, page-cache behavior, CPU transforms, NUMA placement, or the host-to-device copy. A memory-mapped dataset changes the API, but pages still fault into memory as they are touched. GPUDirect Storage can create a more direct DMA path for supported configurations, but it is not an automatic property of every read.

For a well-pipelined steady state, a useful lower bound is:

```text
step time >= max(
  storage or network delivery,
  CPU decode and transforms,
  host-to-device transfer,
  GPU compute
)
```

If the stages do not overlap, their times add instead. If they overlap imperfectly, the result falls between those two cases. This is why increasing GPU compute speed can expose a data bottleneck that was already present but previously hidden.

## Linux execution: processes, threads, and resource domains

### A PID is often a thread group

Linux commonly represents what users call a process as a **thread group**. `/proc/<pid>/status` reports a thread-group ID (`Tgid`), a task ID (`Pid`), and the number of threads (`Threads`). A single application PID therefore does not imply single-threaded execution ([Linux man-pages: `proc_pid_status(5)`](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)).

That distinction matters for AI workloads because one Python process can have:

- Python interpreter threads;
- BLAS or OpenMP worker pools;
- networking threads;
- data-loader child processes;
- CUDA runtime or framework helper threads; and
- asynchronous logging or telemetry workers.

Inspect threads and their current CPUs rather than relying only on process-level CPU:

```bash
PID=12345
ps -eLo pid,tid,ppid,psr,stat,pcpu,pmem,rss,vsz,comm --sort=-pcpu
top -b -n 1 -H -p "$PID"
```

Use batch mode for observation: interactive `top` also offers process-manipulation commands ([procps-ng: `top(1)`](https://man7.org/linux/man-pages/man1/top.1.html)).

The process state is a clue, not a diagnosis. For example, `D` denotes an uninterruptible wait ([Linux kernel: `/proc` filesystem](https://docs.kernel.org/filesystems/proc.html)). It often appears around I/O waits, but does not identify the responsible device or kernel path.

### Forking has AI-specific consequences

`fork()` gives the child a separate virtual address space initially backed by copy-on-write pages. If the parent is multithreaded, only the thread that called `fork()` exists in the child immediately afterward ([Linux man-pages: `fork(2)`](https://man7.org/linux/man-pages/man2/fork.2.html)).

This is relevant to multiprocess data loading. Large Python objects inherited or recreated by workers can multiply host-memory use. Initialization order also matters when processes interact with thread pools, open connections, or accelerator runtimes. Follow the framework's supported multiprocessing guidance rather than assuming every parent state is safe to inherit.

### The container limit may matter more than the host

Cgroups define the effective resource domain for many containerized jobs. Under cgroup v2:

- `memory.current` reports current memory use for the cgroup;
- `memory.high` is a reclaim and throttling boundary;
- `memory.max` is a hard limit; and
- `memory.events` records events such as `high`, `oom`, and `oom_kill`.

These semantics are documented in the Linux kernel's [Control Group v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) guide. A job can therefore be OOM-killed while the physical host still has memory available.

On a cgroup v2 system, a read-only inspection looks like:

```bash
PID=12345
CGROUP_PATH=$(awk -F: '$1 == "0" { print $3 }' "/proc/$PID/cgroup")
CGROUP_DIR="/sys/fs/cgroup${CGROUP_PATH}"

printf 'cgroup: %s\n' "$CGROUP_DIR"
cat "$CGROUP_DIR/memory.current"
cat "$CGROUP_DIR/memory.max"
cat "$CGROUP_DIR/memory.events"
cat "$CGROUP_DIR/cpu.stat"
```

The layout differs on cgroup v1 or hybrid hosts. Inspect `/proc/<pid>/cgroup` first instead of assuming a path.

## Virtual memory is not physical memory

Every process operates in a virtual address space. The kernel maps virtual pages to physical memory, files, shared mappings, devices, or no physical page at all until the process touches the address.

### Read the memory numbers by category

| Signal | What it means | Common mistake |
|---|---|---|
| `VmSize` / VSZ | Total virtual address space | Treating it as RAM consumed |
| `VmRSS` / RSS | Resident mapped pages | Summing RSS across processes despite shared pages |
| `RssAnon` | Resident anonymous memory | Ignoring worker duplication or heap growth |
| `RssFile` | Resident file-backed pages | Treating reclaimable file cache as an application leak |
| `RssShmem` | Resident shared-memory pages | Missing shared-memory queues or tmpfs use |
| `VmSwap` | Process anonymous memory currently in swap | Assuming zero swap means zero pressure |
| `VmPin` | Memory pinned by the process | Forgetting that pinned pages reduce reclaim flexibility |

Linux exposes these fields in `/proc/<pid>/status`; kernel documentation notes that some RSS accounting is asynchronous and can be imprecise. `/proc/<pid>/smaps` is more detailed but expensive, while `smaps_rollup` provides a useful aggregate ([Linux man-pages: `proc_pid_status(5)`](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html); [Linux kernel: `/proc` filesystem](https://docs.kernel.org/filesystems/proc.html)).

```bash
PID=12345
grep -E '^(Name|Pid|Tgid|Threads|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmSwap|VmPin):' \
  "/proc/$PID/status"
cat "/proc/$PID/smaps_rollup"
```

Do not poll `smaps` across thousands of processes at high frequency; reading it requires the kernel to walk mappings.

### Anonymous pages and file-backed pages behave differently

Heap, stack, and anonymous mappings are **anonymous memory**. Executables, shared libraries, and mapped dataset files are **file-backed**. File-backed pages commonly participate in the page cache. The kernel can discard clean file-backed cache pages and read them again later; reclaiming anonymous memory generally requires swap or process termination.

The Linux kernel's [memory-management concepts](https://docs.kernel.org/admin-guide/mm/concepts.html) describe reclaim and the distinction between anonymous and file-backed memory. This is the foundation for understanding why a host can show little `MemFree` and still be healthy.

Use:

```bash
free -h
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|Writeback):' \
  /proc/meminfo
```

`MemAvailable` estimates how much memory can be made available for new work without swapping, including reclaimable cache ([Linux man-pages: `proc_meminfo(5)`](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html)). It is usually a better first signal than `MemFree`.

### Page faults explain when mappings become real work

Creating a mapping with `mmap()` does not necessarily read the whole file. Pages are populated as they are accessed. `MAP_PRIVATE` uses copy-on-write semantics for modifications; `MAP_SHARED` makes changes visible through the shared mapping and, for file mappings, subject to filesystem writeback semantics ([Linux man-pages: `mmap(2)`](https://man7.org/linux/man-pages/man2/mmap.2.html)).

A **minor page fault** is resolved without storage I/O. A **major page fault** requires I/O activity ([Linux man-pages: `getrusage(2)`](https://man7.org/linux/man-pages/man2/getrusage.2.html)). Major-fault growth during startup or an epoch transition is evidence to correlate with cache misses and storage metrics, not proof by itself that a particular disk is slow.

```bash
vmstat 1
pidstat -r -u -d -p "$PID" 1
```

Ignore the first `vmstat` row when looking for current rates; it contains averages since boot, whereas subsequent reports cover the requested sampling interval ([procps-ng: `vmstat(8)`](https://man7.org/linux/man-pages/man8/vmstat.8.html)). Watch subsequent intervals for runnable tasks, blocked tasks, swap-in/out, CPU wait, and context switching.

### Swap and OOM are different failure modes

Swap use alone does not prove a machine is failing. Cold anonymous pages may remain swapped while the active working set runs normally. The damaging case for latency-sensitive AI work is active swap-in or direct reclaim on a critical process.

Conversely, a job can be killed without host-wide swap exhaustion:

- a cgroup can cross `memory.max`;
- a cgroup-local OOM can choose a victim;
- pinned memory and duplicated worker state can reduce reclaim options; or
- a short allocation spike can exceed a hard limit.

Check the cgroup's `memory.events`, process RSS composition, `VmSwap`, fault rate, and logs together.

## NUMA: one server is several neighborhoods

On a NUMA machine, memory-access latency depends on which CPU node owns the memory. Default allocation is generally local to the CPU that first faults the page, often called **first touch** behavior. A NUMA policy primarily affects future allocations; applying a policy after buffers are populated may be too late unless pages are explicitly migrated ([Linux kernel: NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)).

<figure>
  <img src="{{ '/assets/images/linux-networking-storage/numa-device-topology.svg' | relative_url }}" alt="Two-socket NUMA server showing local memory, GPUs, NIC, and NVMe devices attached to different PCIe roots, with a slower cross-socket path.">
  <figcaption>Figure 2. Original diagram by the author, informed by Linux NUMA documentation and NVIDIA topology concepts. Conceptual and not to scale; actual systems vary.</figcaption>
</figure>

GPU numbering does not guarantee proximity to a CPU, NIC, or NVMe device. A transfer may cross:

1. a GPU's PCIe link;
2. a PCIe switch or root complex;
3. a CPU socket's I/O fabric;
4. the inter-socket link; and
5. another socket's memory controller or PCIe root.

Map the actual host:

```bash
lscpu -e
numactl --hardware
lspci -tv
nvidia-smi topo -m
```

Then inspect process placement:

```bash
PID=12345
taskset -pc "$PID"
cat "/proc/$PID/numa_maps"
numactl --show
```

`nvidia-smi topo -m` reports NVIDIA's view of GPU and interconnect relationships; labels and detail vary by platform and driver ([NVIDIA: `nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/index.html)). Correlate it with PCI addresses and `/sys/bus/pci/devices/*/numa_node`.

NUMA problems often present as asymmetry: one rank, worker group, or GPU is slower despite identical code. Before changing affinity, compare CPU sets, memory placement, device locality, and cgroup limits across ranks.

## Networking: resolve, connect, queue, transfer

### DNS is part of request latency

Before a new connection, an application may need to resolve a name. `getaddrinfo()` can return multiple IPv4 or IPv6 addresses and socket choices ([Linux man-pages: `getaddrinfo(3)`](https://man7.org/linux/man-pages/man3/getaddrinfo.3.html)). A fast `dig` result proves only that the queried resolver answered that query; it does not prove the application used the same resolver, address family, search path, or endpoint.

Useful checks, if the tools are already available:

```bash
getent ahosts storage.example.internal
dig +stats storage.example.internal
cat /etc/resolv.conf
```

`dig` sends a network request. Use it deliberately and do not treat it as passive inspection.

### TCP is a reliable byte stream, not a message protocol

TCP provides a reliable, ordered, full-duplex byte stream. It retransmits lost data, but it does not preserve application message boundaries ([Linux man-pages: `tcp(7)`](https://man7.org/linux/man-pages/man7/tcp.7.html); [IETF RFC 9293](https://www.rfc-editor.org/rfc/rfc9293)).

Three concepts matter immediately:

1. **Latency:** the delay before useful bytes arrive.
2. **Bandwidth:** the maximum transfer rate of the path.
3. **Bytes in flight:** unacknowledged data currently traversing the path.

The bandwidth-delay product gives a lower bound for the in-flight data needed to fill a path:

```text
bytes in flight ~= sustained bytes per second x round-trip time
```

For example, a 100 Gb/s path with a 1 ms round-trip time has a bandwidth-delay product of approximately 12.5 MB. A flow with a much smaller effective window cannot fill that path, even though the physical link is fast. Linux supports TCP window scaling, but socket and system buffer limits still matter ([Linux man-pages: `tcp(7)`](https://man7.org/linux/man-pages/man7/tcp.7.html)).

Loss also matters. TCP congestion control reduces sending behavior after signs of congestion; retransmissions add delay and can damage synchronized distributed workloads or tail latency without making aggregate link utilization look high ([IETF RFC 5681](https://www.rfc-editor.org/rfc/rfc5681)).

### Queues can fill before bandwidth does

A server's `listen()` backlog is the queue of fully established connections waiting for the application to call `accept()` on modern Linux. Incomplete handshakes use a separate SYN queue, and `somaxconn` caps the requested accept backlog ([Linux man-pages: `listen(2)`](https://man7.org/linux/man-pages/man2/listen.2.html)).

For model servers, object-store gateways, and metadata services, latency can therefore rise because:

- DNS or connection setup repeats too often;
- the accept queue fills;
- application workers do not drain established connections;
- socket buffers limit a long-fat network path;
- packets are dropped or retransmitted; or
- serialization and decompression, not the network, consume the time.

### Read counters as deltas

```bash
ss -s
ss -ltn
ss -tanpi
ip -s link show dev eth0
ethtool eth0
ethtool -S eth0
nstat -a
```

Take two snapshots around the same workload interval. `tx_bytes` means bytes handed to the device, not proof that a peer received them. Interface drops and errors should be correlated with TCP retransmissions and application latency ([Linux kernel: interface statistics](https://docs.kernel.org/networking/statistics.html)).

Counter names are driver-specific, and some tools require elevated permission for particular details. Do not reset counters during an investigation.

## Storage: semantics first, performance second

The word “storage” hides three different abstractions:

| Abstraction | Interface | What it provides | Typical AI use |
|---|---|---|---|
| Filesystem | Paths, directories, file descriptors | Naming, permissions, metadata, file semantics | Checkpoints, datasets, logs, shared training trees |
| Block device | Addressable blocks | Raw storage presented to a filesystem or database | Local NVMe, attached volumes, database backing |
| Object storage | Key/object API | Large immutable or replaceable objects through service APIs | Dataset shards, model artifacts, checkpoints, archives |

Mounting an object-store gateway does not automatically make object semantics identical to a local POSIX filesystem. Metadata operations, consistency behavior, rename semantics, caching, and failure modes depend on the implementation.

Ceph makes the distinction concrete: CephFS is a POSIX filesystem, RBD is a block interface, and RGW is an object gateway. CephFS separately serves metadata through Metadata Servers, which helps explain why many-small-file workloads can behave differently from large sequential reads ([Ceph Foundation: CephFS](https://docs.ceph.com/en/latest/cephfs/)).

### Buffered I/O and the page cache

Normal file reads usually use the page cache:

1. The process calls `read()` or touches a mapped page.
2. If the page is cached, the kernel can satisfy the request from memory.
3. If not, the kernel schedules storage I/O and later places the page in cache.
4. Clean cached pages can be reclaimed under memory pressure.

This produces a classic benchmark trap: epoch one is cold, while epoch two is warm. The application did not become faster; the workload changed from storage reads to memory hits.

`/proc/<pid>/io` helps separate the layers:

- `rchar` counts bytes returned by read-like system calls;
- `read_bytes` counts bytes actually fetched from storage for block-backed files.

If `rchar` rises quickly while `read_bytes` barely moves, the process may be reading cached data. The exact accounting and permissions are documented in [`proc_pid_io(5)`](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html).

### `mmap` and direct I/O are not magic speed switches

`mmap()` can simplify random access and avoid an explicit user-space `read()` loop, but the kernel still services page faults and manages cache pages.

`O_DIRECT` requests reduced page-cache effects, but alignment and behavior depend on the filesystem and device. The Linux `open(2)` manual explicitly warns that direct-I/O restrictions vary; it is not a portable “make I/O faster” flag ([Linux man-pages: `open(2)`](https://man7.org/linux/man-pages/man2/open.2.html)).

Direct I/O can be useful when an application has its own cache or must avoid double caching. It can also make small or misaligned access slower and more complex. Measure the intended workload rather than choosing an I/O mode by name.

### Throughput, IOPS, and latency answer different questions

- **Throughput:** bytes transferred per second.
- **IOPS:** operations completed per second.
- **Latency:** time for one operation, usually summarized with percentiles.
- **Queue depth:** outstanding requests available for the device or service to process.

A workload can exhaust IOPS with 4 KiB random reads while using little bandwidth. Another can stream multi-megabyte shards at high bandwidth with relatively few operations. A third can have strong averages but unacceptable p99 latency.

Always report:

- request or shard size;
- sequential versus random access;
- read/write mix;
- concurrency and queue depth;
- warm or cold cache state;
- compression and CPU transforms;
- mean and tail latency; and
- whether the metric comes from the process, kernel block layer, client, or remote service.

### Small files are often a metadata workload

One sample per file may require path traversal, permission checks, inode and directory lookups, open/close operations, and small reads. Adding workers can increase metadata pressure rather than storage throughput.

Packing samples into larger immutable shards can reduce open and metadata operations and improve sequential access, but it changes:

- shuffle granularity;
- retry behavior;
- partial-read cost;
- update and invalidation strategy; and
- cache placement.

The correct shard size is a workload decision, not a universal constant.

## From a dataset to GPU memory

PyTorch's `DataLoader` supports single- and multiprocess loading, batching, prefetching, and pinned-memory transfer. Map-style datasets fit indexed access; iterable datasets fit streams and sources where random access is expensive. Multiprocess loading can duplicate Python objects, and iterable datasets must be sharded correctly to avoid duplicate samples ([PyTorch: `torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)).

<figure>
  <img src="{{ '/assets/images/linux-networking-storage/ai-input-pipeline.svg' | relative_url }}" alt="Overlapped AI input pipeline showing storage or network delivery, CPU decode and transforms, pinned-memory staging, host-to-device transfer, and GPU compute.">
  <figcaption>Figure 3. Original diagram by the author, based on the staged data path described by PyTorch and NVIDIA CUDA documentation. Conceptual and not to scale.</figcaption>
</figure>

### Worker count is a pipeline control

More workers can help when decode or I/O latency is parallelizable. More workers can hurt when they:

- duplicate a large parent dataset object;
- contend for CPU or memory bandwidth;
- produce too many small storage requests;
- overload a metadata service;
- increase page-cache churn;
- duplicate iterable-dataset samples; or
- fill prefetch queues with batches the GPU cannot consume.

Tune worker count against end-to-end step time, not worker CPU utilization alone.

### Pinned memory trades flexibility for transfer efficiency

Page-locked host memory allows the CUDA runtime to perform efficient asynchronous host-to-device transfers under the right conditions. PyTorch exposes this through `pin_memory`, while NVIDIA documents page-locked host memory in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

Pinned memory is not free:

- the kernel cannot reclaim it like ordinary pageable memory;
- large prefetch queues can pin many batches;
- multiprocess workers can amplify host-memory use; and
- a container can hit its cgroup limit before the host is full.

Measure pinned-memory volume, cgroup events, copy overlap, and step time together.

### GPUDirect Storage changes the path, not the need to measure

NVIDIA GPUDirect Storage provides a supported DMA path between storage and GPU memory intended to avoid a CPU bounce buffer. It can reduce CPU load and transfer overhead, but it is conditional on the GPU, driver, filesystem or block path, mount, and system configuration.

NVIDIA also documents a compatibility path that can fall back to POSIX I/O through host memory when direct operation is unavailable. “The package is installed” therefore does not prove the direct path is active. GDS calls are still issued by the CPU; “direct” describes the data-transfer path ([NVIDIA: GPUDirect Storage cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)).

## Build a simple performance budget

Before tuning, calculate the rate the pipeline must sustain.

### Example 1: input throughput

Suppose each step consumes 1 GiB of source data and the target step time is 0.5 seconds:

```text
required source throughput = 1 GiB / 0.5 s = 2 GiB/s
```

That is a floor, not a procurement target. Add headroom for:

- tail latency;
- competing jobs;
- cache misses;
- retries;
- metadata;
- uneven shard sizes; and
- startup or checkpoint traffic.

Also distinguish **source bytes** from **decoded bytes**. Compression can reduce storage and network bytes while increasing CPU work and memory bandwidth.

### Example 2: high-bandwidth network path

For a 100 Gb/s path and 1 ms RTT:

```text
100 Gb/s / 8 = 12.5 GB/s
12.5 GB/s x 0.001 s = 12.5 MB in flight
```

This does not mean every application should manually set a 12.5 MB socket buffer. It means the end-to-end protocol and TCP windows must permit roughly that order of in-flight data to sustain the rate. Connection count, request size, congestion control, kernel limits, and remote service behavior all affect the result.

### Example 3: the small-file tax

A loader consuming 10,000 samples per second from one file per sample requires at least 10,000 file opens and reads per second before accounting for directory traversal, attributes, retries, or shuffle. If those samples are packed into ten sequential shards per second, the byte rate can be identical while metadata operations differ by orders of magnitude.

This is why “the storage system supports 5 GB/s” is not enough information.

## A read-only diagnostic workflow

The goal is to collect one synchronized interval around a reproducible slow period. Commands marked “if available” should only be used when already approved and installed. Some read-only commands can still create load on large shared systems.

### 1. Scope the process and topology

```bash
PID=12345

uname -a
lscpu
lscpu -e
numactl --hardware            # if available
lspci -tv                     # if available
nvidia-smi
nvidia-smi topo -m

cat "/proc/$PID/cgroup"
taskset -pc "$PID"
```

Record the process or rank, container/cgroup, GPU, NIC, dataset path, mount, and underlying device.

### 2. Inspect process and cgroup memory

```bash
grep -E '^(Name|Pid|Tgid|Threads|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmSwap|VmPin):' \
  "/proc/$PID/status"
cat "/proc/$PID/smaps_rollup"
cat "/proc/$PID/io"
cat "/proc/$PID/numa_maps"

free -h
vmstat 1
pidstat -r -u -d -p "$PID" 1  # if available
```

For cgroup v2:

```bash
CGROUP_PATH=$(awk -F: '$1 == "0" { print $3 }' "/proc/$PID/cgroup")
CGROUP_DIR="/sys/fs/cgroup${CGROUP_PATH}"

grep -H . \
  "$CGROUP_DIR/memory.current" \
  "$CGROUP_DIR/memory.max" \
  "$CGROUP_DIR/memory.events" \
  "$CGROUP_DIR/cpu.stat"
```

### 3. Inspect network state as deltas

```bash
ss -s
ss -tanpi
ip -s link show dev eth0
ethtool eth0             # if available and permitted
ethtool -S eth0          # if available and permitted
nstat -a                 # if available
```

Capture the same commands before and after the slow interval. Correlate retransmissions, drops, errors, socket queue growth, and connection states with application latency.

### 4. Map the storage path

```bash
DATASET=/path/to/dataset

findmnt -T "$DATASET"
df -hT "$DATASET"
df -i "$DATASET"
lsblk -o NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINTS,MODEL
iostat -x -y 1           # if available
```

`df` reports space and inode figures for the filesystem containing a path, while `du` recursively summarizes space attributed to directory entries ([GNU Coreutils: `df` invocation](https://www.gnu.org/software/coreutils/manual/html_node/df-invocation.html); [GNU Coreutils: `du` invocation](https://www.gnu.org/software/coreutils/manual/html_node/du-invocation.html)). Recursive `du` or `find` over a large shared dataset can create substantial metadata load; run them only with approval.

Do not run an ordinary `fio` workload against a production path or raw device. `fio` is a workload generator, not passive inspection. Even a read-only test consumes cache, device bandwidth, CPU, and service capacity ([fio documentation](https://fio.readthedocs.io/en/latest/fio_doc.html)).

### 5. Correlate with the GPU timeline

```bash
nvidia-smi dmon -s pucvmet -c 10
nvidia-smi pmon -c 1
```

Availability and supported fields vary by GPU and driver. The important comparison is timing:

- Do GPU idle gaps line up with blocked workers?
- Do major faults or storage latency spike first?
- Does CPU transform time fill the gap?
- Does one rank differ because of topology or cache state?
- Does host-to-device copy overlap with compute?

GPU utilization is the symptom boundary, not the root-cause label.

## Common production patterns

| Symptom | Plausible causes | Evidence to collect |
|---|---|---|
| GPU utilization has regular idle gaps | Loader starvation, synchronous copies, decode bottleneck, small reads | Step timeline, worker CPU, faults, `/proc/<pid>/io`, storage latency |
| Job OOM-killed while host has available RAM | Cgroup hard limit, worker duplication, pinned queues, allocation spike | `memory.max`, `memory.events`, RSS composition, `VmPin` |
| First epoch is slow; later epochs are fast | Page-cache warming, local cache population, remote metadata caching | `rchar` versus `read_bytes`, device reads, controlled cache-state comparison |
| Only one rank is slow | NUMA/PCIe mismatch, CPU-set difference, remote cache miss, network path | `numa_maps`, topology, cgroup limits, routes, counters per node |
| Network throughput is high but steps are slow | Retransmits, RTT/window limit, small requests, serialization, metadata | `ss -ti`, `nstat`, NIC deltas, request size and latency percentiles |
| More loader workers make performance worse | Metadata storm, cache churn, CPU contention, object duplication | Open/read rate, worker RSS, storage latency, CPU run queue |
| Storage reports high MB/s but application stalls | IOPS or tail-latency limit, queueing, CPU decode, cache misses | Request size, p95/p99 latency, queue depth, CPU profile |

## Change one variable at a time

After establishing a baseline, change one control:

- data-loader worker count;
- prefetch depth;
- batch size;
- shard size;
- compression level;
- local-cache placement;
- CPU or memory affinity;
- connection reuse;
- request concurrency; or
- copy overlap.

Repeat the same workload interval and compare the same signals. A faster warm-cache run does not validate a storage change if the baseline was cold. A higher aggregate throughput result does not validate a latency-sensitive serving change if p99 regressed.

## Decision checklist

Before declaring an AI workload compute-, network-, or storage-bound, answer:

1. What exact interval is slow: startup, first epoch, steady state, checkpoint, or recovery?
2. Is the GPU continuously busy, or are there bubbles between kernels?
3. What are the process and cgroup limits?
4. How much memory is anonymous, file-backed, shared, swapped, and pinned?
5. Are major faults, reclaim, or swap-ins rising during the slow interval?
6. Which NUMA node owns the process memory?
7. Which PCIe roots contain the GPU, NIC, and NVMe device?
8. Is the source a filesystem, block-backed mount, object API, or gateway?
9. Is the access pattern large/sequential or small/random?
10. Are measurements cold-cache, warm-cache, or mixed?
11. What are the request size, concurrency, throughput, IOPS, and p95/p99 latency?
12. Are TCP retransmissions, drops, or socket queues increasing?
13. Is CPU decode, tokenization, augmentation, or decompression the real bottleneck?
14. Is pinned-memory volume bounded?
15. Does the proposed change improve end-to-end step time or goodput under the target SLO?

## Sources, attribution, and diagrams

This article paraphrases technical documentation for explanation. All three diagrams are original, conceptual illustrations by the author; no source figures, screenshots, or tables are reproduced. Product and project names belong to their respective owners.

Where command behavior or a non-obvious mechanism is introduced, the surrounding text links to the relevant Linux kernel documentation, Linux man-pages entry, RFC, or official project/vendor documentation. Benchmark examples are illustrative calculations, not measured vendor results.

License note: this article reproduces no source prose, code, tables, screenshots, or figures. Any future reuse must be checked against the source-specific terms: Linux documentation uses per-file [SPDX licensing rules](https://docs.kernel.org/process/license-rules.html), NVIDIA documentation is subject to [NVIDIA's legal terms](https://www.nvidia.com/en-us/about-nvidia/legal-info/), and project documentation may carry repository-specific licenses.

## References

1. **Linux kernel documentation contributors.** [*Control Group v2*](https://docs.kernel.org/admin-guide/cgroup-v2.html). Linux Kernel Documentation.
2. **Linux kernel documentation contributors.** [*Linux Memory Management Concepts*](https://docs.kernel.org/admin-guide/mm/concepts.html). Linux Kernel Documentation.
3. **Linux kernel documentation contributors.** [*NUMA Memory Policy*](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html). Linux Kernel Documentation.
4. **Linux kernel documentation contributors.** [*The `/proc` Filesystem*](https://docs.kernel.org/filesystems/proc.html). Linux Kernel Documentation.
5. **Linux kernel documentation contributors.** [*Interface statistics*](https://docs.kernel.org/networking/statistics.html). Linux Kernel Documentation.
6. **Linux man-pages project.** [*proc_pid_status(5)*](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html): process status and memory fields.
7. **Linux man-pages project.** [*proc_meminfo(5)*](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html): system memory information.
8. **Linux man-pages project.** [*proc_pid_io(5)*](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html): per-process I/O statistics.
9. **Linux man-pages project.** [*fork(2)*](https://man7.org/linux/man-pages/man2/fork.2.html): create a child process.
10. **Linux man-pages project.** [*mmap(2)*](https://man7.org/linux/man-pages/man2/mmap.2.html): map files or devices into memory.
11. **Linux man-pages project.** [*open(2)*](https://man7.org/linux/man-pages/man2/open.2.html): file opening and `O_DIRECT`.
12. **Linux man-pages project.** [*getrusage(2)*](https://man7.org/linux/man-pages/man2/getrusage.2.html): resource usage and page faults.
13. **Linux man-pages project.** [*listen(2)*](https://man7.org/linux/man-pages/man2/listen.2.html): socket listen queues.
14. **Linux man-pages project.** [*tcp(7)*](https://man7.org/linux/man-pages/man7/tcp.7.html): Linux TCP behavior and controls.
15. **Linux man-pages project.** [*getaddrinfo(3)*](https://man7.org/linux/man-pages/man3/getaddrinfo.3.html): network address and service translation.
16. **IETF.** [*RFC 9293: Transmission Control Protocol*](https://www.rfc-editor.org/rfc/rfc9293).
17. **Allman, M.; Paxson, V.; Blanton, E. / IETF.** [*RFC 5681: TCP Congestion Control*](https://www.rfc-editor.org/rfc/rfc5681).
18. **NVIDIA.** [*CUDA Programming Guide*](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
19. **NVIDIA.** [*NVIDIA System Management Interface (`nvidia-smi`)*](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
20. **NVIDIA.** [*GPUDirect Storage cuFile API Reference Guide*](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html).
21. **PyTorch contributors.** [*torch.utils.data*](https://docs.pytorch.org/docs/stable/data.html). PyTorch documentation.
22. **Ceph Foundation and contributors.** [*Ceph File System*](https://docs.ceph.com/en/latest/cephfs/).
23. **Axboe, J.; Fu, V.; and fio contributors.** [*fio — Flexible I/O Tester Documentation*](https://fio.readthedocs.io/en/latest/fio_doc.html).
24. **Linux kernel documentation contributors.** [*Linux kernel licensing rules*](https://docs.kernel.org/process/license-rules.html).
25. **procps-ng contributors.** [*top(1): display Linux processes*](https://man7.org/linux/man-pages/man1/top.1.html).
26. **procps-ng contributors.** [*vmstat(8): report virtual memory statistics*](https://man7.org/linux/man-pages/man8/vmstat.8.html).
27. **GNU Project.** [*GNU Coreutils: `df` invocation*](https://www.gnu.org/software/coreutils/manual/html_node/df-invocation.html).
28. **GNU Project.** [*GNU Coreutils: `du` invocation*](https://www.gnu.org/software/coreutils/manual/html_node/du-invocation.html).
29. **NVIDIA.** [*Legal Information*](https://www.nvidia.com/en-us/about-nvidia/legal-info/).

All links were accessed on July 29, 2026.

## Changelog

- **2026-07-29:** Initial publication.
