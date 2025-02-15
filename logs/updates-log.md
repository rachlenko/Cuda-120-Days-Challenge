## Day 1: Introduction to GPU Computing & CUDA
- **Commit:** 355bba6
- **Description:** Added Day 1 notes and project setup.
- **Timestamp:** 2025-01-24 02:07:57

## Day 2: Setting Up the Development Environment
- **Commit:** 355bba6
- **Description:** Installed CUDA Toolkit and ran sample codes.
- **Timestamp:** 2025-01-24 02:07:57

## Day 3: GPU vs. CPU Architecture Foundations
- **Commit:** 355bba6
- **Description:** Compared GPU SMs and CPU cores.
- **Timestamp:** 2025-01-24 02:07:57

## Day 4: Thread Hierarchy: Grids & Blocks
- **Commit:** 355bba6
- **Description:** Launched kernels with different grid/block dimensions.
- **Timestamp:** 2025-01-24 02:07:57
## Day 1: Introduction to GPU Computing & CUDA
- **Commit:** e1a9d27
- **Description:** Added Day 1 notes and project setup.
- **Timestamp:** 2025-01-24 02:55:11

## Day 3: GPU vs. CPU Architecture Foundations
- **Commit:** e1a9d27
- **Description:** Compared GPU SMs and CPU cores.
- **Timestamp:** 2025-01-24 02:55:11

## Day 4: Thread Hierarchy: Grids & Blocks
- **Commit:** e1a9d27
- **Description:** Launched kernels with different grid/block dimensions.
- **Timestamp:** 2025-01-24 02:55:11

## Day 8: Memory Allocation & Pointers
- **Commit:** e1a9d27
- **Description:** Used cudaMalloc/cudaFree; practiced error checking.
- **Timestamp:** 2025-01-24 02:55:11

## Day 9: Memory Alignment & Coalescing
- **Commit:** e1a9d27
- **Description:** Benchmarked coalesced vs. non-coalesced memory accesses.
- **Timestamp:** 2025-01-24 02:55:11

## Day 10: Shared Memory Fundamentals
- **Commit:** e1a9d27
- **Description:** Implemented tile-based matrix multiplication using shared memory.
- **Timestamp:** 2025-01-24 02:55:11

## Day 11: Thread Synchronization (__syncthreads())
- **Commit:** e1a9d27
- **Description:** Extended tile-based multiplication with sync calls.
- **Timestamp:** 2025-01-24 02:55:11

## Day 12: Bank Conflicts in Shared Memory
- **Commit:** e1a9d27
- **Description:** Tested access patterns causing bank conflicts.
- **Timestamp:** 2025-01-24 02:55:11

## Day 13: Basic Atomic Operations
- **Commit:** e1a9d27
- **Description:** Used atomicAdd to sum an array in parallel.
- **Timestamp:** 2025-01-24 02:55:11

## Day 14: Progress Checkpoint
- **Commit:** e1a9d27
- **Description:** Quick recap or quiz: global vs. shared memory usage.
- **Timestamp:** 2025-01-24 02:55:11

## Day 15: Advanced Atomic Operations
- **Commit:** e1a9d27
- **Description:** Experimented with atomicCAS, atomicExch, etc.
- **Timestamp:** 2025-01-24 02:55:11

## Day 16: Kernel Configuration Tuning
- **Commit:** e1a9d27
- **Description:** Adjusted block sizes for the same kernel.
- **Timestamp:** 2025-01-24 02:55:11

## Day 17: Host-Device Synchronization Patterns
- **Commit:** e1a9d27
- **Description:** Used cudaDeviceSynchronize() for timing.
- **Timestamp:** 2025-01-24 02:55:11

## Day 18: Error Handling & cudaGetErrorString()
- **Commit:** e1a9d27
- **Description:** Implemented robust error checks after each CUDA call.
- **Timestamp:** 2025-01-24 02:55:11

## Day 19: Unified Memory (UM) Intro
- **Commit:** e1a9d27
- **Description:** Used cudaMallocManaged; ran simple vector addition.
- **Timestamp:** 2025-01-24 02:55:11

## Day 20: Capstone Project #1
- **Commit:** e1a9d27
- **Description:** Implemented 2D convolution (edge detection) on the GPU.
- **Timestamp:** 2025-01-24 02:55:11

## Day 21: Streams & Concurrency (Basics)
- **Commit:** e1a9d27
- **Description:** Launched two kernels in different streams.
- **Timestamp:** 2025-01-24 02:55:11

## Day 22: Events & Timing
- **Commit:** e1a9d27
- **Description:** Used CUDA events for precise kernel timing.
- **Timestamp:** 2025-01-24 02:55:11

## Day 23: Asynchronous Memory Copy
- **Commit:** e1a9d27
- **Description:** Copied data using streams asynchronously.
- **Timestamp:** 2025-01-24 02:55:11

## Day 24: Pinned (Page-Locked) Memory
- **Commit:** e1a9d27
- **Description:** Compared pinned vs. pageable host memory transfers.
- **Timestamp:** 2025-01-24 02:55:11

## Day 25: Double Buffering Technique
- **Commit:** e1a9d27
- **Description:** Implemented a two-buffer pipeline to overlap compute and transfer.
- **Timestamp:** 2025-01-24 02:55:11

## Day 26: Constant Memory
- **Commit:** e1a9d27
- **Description:** Used constant memory for read-only data.
- **Timestamp:** 2025-01-24 02:55:11

## Day 27: Texture & Surface Memory (Intro)
- **Commit:** e1a9d27
- **Description:** Sampled a small 2D texture; compared vs. global memory fetch.
- **Timestamp:** 2025-01-24 02:55:11

## Day 28: Progress Checkpoint
- **Commit:** e1a9d27
- **Description:** Recap concurrency & memory (short quiz or multi-topic mini-project).
- **Timestamp:** 2025-01-24 02:55:11

## Day 29: Texture Memory (Practical)
- **Commit:** e1a9d27
- **Description:** Implemented image-processing kernel (e.g., grayscale) using textures.
- **Timestamp:** 2025-01-24 02:55:11

## Day 30: Surface Memory
- **Commit:** e1a9d27
- **Description:** Wrote operations using surfaces (e.g., output image buffer).
- **Timestamp:** 2025-01-24 02:55:11

## Day 31: Unified Memory Deep Dive
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used cudaMallocManaged with multiple kernels; measured page-fault overhead.
- **Timestamp:** $(date +"%F %T")

## Day 32: Stream Sync & Dependencies
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Enforced execution order with events or cudaStreamWaitEvent().
- **Timestamp:** $(date +"%F %T")

## Day 33: Intro to CUDA Graphs
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Converted a kernel sequence into a CUDA graph; measured performance.
- **Timestamp:** $(date +"%F %T")

## Day 34: Nsight Systems / Nsight Compute
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Profiled a small app to find bottlenecks; read kernel timelines.
- **Timestamp:** $(date +"%F %T")

## Day 35: Occupancy & Launch Config Tuning
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used the Occupancy Calculator to refine block size for better SM use.
- **Timestamp:** $(date +"%F %T")

## Day 36: Profiling & Bottleneck Analysis
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Profiled matrix multiplication or similar; identified memory vs. compute limits.
- **Timestamp:** $(date +"%F %T")

## Day 37: Intro to Warp-Level Primitives
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used warp shuffle instructions for a small parallel reduce.
- **Timestamp:** $(date +"%F %T")

## Day 38: Warp Divergence
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Wrote a kernel with branching; measured performance difference.
- **Timestamp:** $(date +"%F %T")

## Day 39: Dynamic Parallelism
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Launched kernels from within a kernel to handle subdivided tasks.
- **Timestamp:** $(date +"%F %T")

## Day 40: Capstone Project #2
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented Sparse Matrix-Vector Multiplication for large sparse data sets.
- **Timestamp:** $(date +"%F %T")

## Day 41: Advanced Streams & Multi-Stream Concurrency
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Launched multiple kernels in parallel using multiple streams.
- **Timestamp:** $(date +"%F %T")

## Day 42: Progress Checkpoint
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Recap concurrency, warp ops, dynamic parallelism.
- **Timestamp:** $(date +"%F %T")

## Day 43: Efficient Data Transfers & Zero-Copy
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Mapped host memory into device space (zero-copy); measured overhead vs. pinned.
- **Timestamp:** $(date +"%F %T")

## Day 44: Advanced Warp Intrinsics (Scan, etc.)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented a warp-wide prefix sum with __shfl_down_sync.
- **Timestamp:** $(date +"%F %T")

## Day 45: Cooperative Groups (Intro)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used cooperative groups for flexible synchronization within blocks or grids.
- **Timestamp:** $(date +"%F %T")

## Day 46: Peer-to-Peer Communication (Multi-GPU)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Enabled P2P for direct data transfers (if you have multiple GPUs).
- **Timestamp:** $(date +"%F %T")

## Day 47: Intermediate Debugging & Profiling Tools
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used cuda-gdb or Nsight Eclipse for step-by-step debugging.
- **Timestamp:** $(date +"%F %T")

## Day 48: Memory Footprint Optimization
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Reduced shared memory or register usage; measured occupancy.
- **Timestamp:** $(date +"%F %T")

## Day 49: Thrust for High-Level Operations
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Replaced custom loops with Thrust transforms, sorts, reductions.
- **Timestamp:** $(date +"%F %T")

## Day 50: Intro to cuBLAS
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Performed basic vector/matrix ops with cuBLAS, compared to custom kernels.
- **Timestamp:** $(date +"%F %T")

## Day 51: Intro to cuFFT
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented a simple 1D FFT on the GPU; measured performance.
- **Timestamp:** $(date +"%F %T")

## Day 52: Code Optimization (Part 1)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Applied loop unrolling or register usage tweaks; measured improvements.
- **Timestamp:** $(date +"%F %T")

## Day 53: Code Optimization (Part 2)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Analyzed PTX, applied instruction-level optimizations.
- **Timestamp:** $(date +"%F %T")

## Day 54: Nsight Compute: Kernel Analysis
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Examined occupancy, memory throughput, and instruction mix.
- **Timestamp:** $(date +"%F %T")

## Day 55: Intro to Device Libraries (cuRAND, etc.)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Generated random numbers (cuRAND); ran a Monte Carlo simulation.
- **Timestamp:** $(date +"%F %T")

## Day 56: Progress Checkpoint
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Recap concurrency (multi-stream), libraries, optimization.
- **Timestamp:** $(date +"%F %T")

## Day 57: Robust Error Handling & Debugging
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Expanded error checking macros; advanced debugging with cuda-gdb.
- **Timestamp:** $(date +"%F %T")

## Day 58: Handling Large Data Sets
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Chunked large arrays with streaming techniques.
- **Timestamp:** $(date +"%F %T")

## Day 59: MPS (Multi-Process Service)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Enabled MPS for sharing GPU among multiple processes (if supported).
- **Timestamp:** $(date +"%F %T")

## Day 60: Capstone Project #3
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented Multi-Stream Data Processing: Overlap transfers & kernels for real-time feeds.
- **Timestamp:** $(date +"%F %T")
