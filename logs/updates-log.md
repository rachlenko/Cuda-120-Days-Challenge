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

## Day 61: GPU-Accelerated Sorting
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used Thrust’s sort; compared vs. CPU for large data.
- **Timestamp:** $(date +"%F %T")

## Day 62: Stream Compaction & Parallel Patterns
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented parallel compaction (remove zeros) via Thrust or custom.
- **Timestamp:** $(date +"%F %T")

## Day 63: Concurrency Patterns (Producer-Consumer)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Pipeline kernels: one generating data, one consuming it.
- **Timestamp:** $(date +"%F %T")

## Day 64: Pinned + Unified Memory Hybrid
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used pinned memory for input streaming, unified memory for intermediate results.
- **Timestamp:** $(date +"%F %T")

## Day 65: Collaborative Grouping Techniques
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used cooperative groups for advanced reductions.
- **Timestamp:** $(date +"%F %T")

## Day 66: Peer-to-Peer (P2P) & Multi-GPU Scaling
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Split data across multiple GPUs if available.
- **Timestamp:** $(date +"%F %T")

## Day 67: GPU-Accelerated Graph Analytics (Intro)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Simple BFS or PageRank with adjacency lists on the GPU.
- **Timestamp:** $(date +"%F %T")

## Day 68: Memory Pool & Custom Allocators
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Reused device memory with a custom allocator to reduce cudaMalloc overhead.
- **Timestamp:** $(date +"%F %T")

## Day 69: Occupancy-Based Tuning for Large Problems
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Maximized occupancy on a large matrix multiplication.
- **Timestamp:** $(date +"%F %T")

## Day 70: Progress Checkpoint
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Recap concurrency patterns, advanced memory, multi-GPU.
- **Timestamp:** $(date +"%F %T")

## Day 71: Advanced Streams & Overlapping
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Overlapped multiple kernels, data transfers, and CPU tasks.
- **Timestamp:** $(date +"%F %T")

## Day 72: CUDA Graphs: Complex Workflows
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Merged dependent kernels & copies into one CUDA graph.
- **Timestamp:** $(date +"%F %T")

## Day 73: Dynamic Graph Launches
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Built and launched graphs at runtime based on conditions.
- **Timestamp:** $(date +"%F %T")

## Day 74: Multi-GPU Programming (Deeper Exploration)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Distributed workload across two GPUs if hardware supports.
- **Timestamp:** $(date +"%F %T")

## Day 75: Performance Metrics & Roofline Analysis
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Collected memory throughput, FLOPS, chart on a roofline.
- **Timestamp:** $(date +"%F %T")

## Day 76: Mixed Precision & Tensor Cores (If Supported)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Implemented half-precision (FP16) matrix multiply on Tensor Cores.
- **Timestamp:** $(date +"%F %T")

## Day 77: UM Advanced Topics (Prefetch, Advise)
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Used cudaMemAdvise, prefetch data to specific devices.
- **Timestamp:** $(date +"%F %T")

## Day 78: Large-Scale Projects: Modular Kernel Design
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Split large kernels into smaller, manageable modules.
- **Timestamp:** $(date +"%F %T")

## Day 79: Portability & Scalability Best Practices
- **Commit:** $(git rev-parse --short HEAD)
- **Description:** Adjusted code for various GPU architectures (SM versions).
- **Timestamp:** $(date +"%F %T")

 ## Day 80: Capstone Project #4
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Implemented Multi-GPU Matrix Multiply: Split large matrix across 2 GPUs.
   - **Timestamp:** $(date +"%F %T")

   ## Day 81: Cooperative Groups: Advanced Patterns
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Tried a grid-level cooperative kernel needing all blocks to sync.
   - **Timestamp:** $(date +"%F %T")

   ## Day 82: Large-Scale Batch Processing
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used batched operations (cuBLAS batched GEMM) for efficiency.
   - **Timestamp:** $(date +"%F %T")

   ## Day 83: External Libraries (cuDNN, etc.)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Integrated a small NN layer using cuDNN if possible.
   - **Timestamp:** $(date +"%F %T")

   ## Day 84: Progress Checkpoint
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Reflect on concurrency, multi-GPU, libraries.
   - **Timestamp:** $(date +"%F %T")

   ## Day 85: Instruction Throughput Profiling
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used Nsight Compute to track instruction throughput for tight kernels.
   - **Timestamp:** $(date +"%F %T")

   ## Day 86: Occupancy vs. ILP
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Compared effects of occupancy vs. ILP (Instruction-Level Parallelism).
   - **Timestamp:** $(date +"%F %T")

   ## Day 87: Custom Memory Allocators
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Extended your memory pool design with stream-ordered allocations.
   - **Timestamp:** $(date +"%F %T")

   ## Day 88: Kernel Fusion & Loop Fusion
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Merged multiple small kernels into a single kernel to reduce launch overhead.
   - **Timestamp:** $(date +"%F %T")

   ## Day 89: Algorithmic Optimizations (Tiling, Blocking)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Refined tiling or blocking for matrix multiply, convolution, etc.
   - **Timestamp:** $(date +"%F %T")

   ## Day 90: Minimizing Data Transfers
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used pinned memory, async transfers, or kernel-side generation to limit PCIe overhead.
   - **Timestamp:** $(date +"%F %T")

   ## Day 91: Enterprise-Level Code Structure
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Explored multi-file, multi-module approach with separate compilation.
   - **Timestamp:** $(date +"%F %T")

   ## Day 92: Advanced Debugging (Races & Deadlocks)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Diagnosed a race or deadlock in a complex multi-stream or multi-block scenario.
   - **Timestamp:** $(date +"%F %T")

   ## Day 93: Real-Time GPU Computing Techniques
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** If real-time constraints exist, explored low-latency execution patterns.
   - **Timestamp:** $(date +"%F %T")

   ## Day 94: Host Multithreading + GPU Coordination
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used multiple CPU threads to launch kernels/manage streams concurrently.
   - **Timestamp:** $(date +"%F %T")

   ## Day 95: CUDA Graph Updates & Reusability
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Dynamically updated parts of a CUDA graph without a full rebuild.
   - **Timestamp:** $(date +"%F %T")

   ## Day 96: Precision & Numerical Stability
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Examined rounding, float vs. double, iterative error accumulation.
   - **Timestamp:** $(date +"%F %T")

   ## Day 97: Advanced P2P & Clustering (If Possible)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used GPU-GPU RDMA or multi-node scaling in a cluster environment.
   - **Timestamp:** $(date +"%F %T")

   ## Day 98: Progress Checkpoint
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Recap advanced debugging, multi-threaded host, graphs, precision.
   - **Timestamp:** $(date +"%F %T")

   ## Day 99: Graph API for Complex DAG Workloads
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Built a multi-kernel DAG with conditional branches/loops using CUDA Graphs.
   - **Timestamp:** $(date +"%F %T")

   ## Day 100: Capstone Project #5
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** CUDA Graph-Optimized Workload: Merge multiple kernels + copies into one graph.
   - **Timestamp:** $(date +"%F %T")

   ## Day 101: GPU-Accelerated ML Frameworks (Intro)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** If possible, integrated a custom kernel/layer into TensorFlow or PyTorch.
   - **Timestamp:** $(date +"%F %T")

   ## Day 102: CUDA + Other Parallel Frameworks
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Explored hybrid CPU/GPU parallelism (OpenMP, MPI).
   - **Timestamp:** $(date +"%F %T")

   ## Day 103: Tuning GPU-Accelerated ML Ops
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Profiled a small neural net or inference pipeline; identified GPU hotspots.
   - **Timestamp:** $(date +"%F %T")

   ## Day 104: Multi-GPU Scaling in ML
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Distributed training across multiple GPUs or data parallel approach.
   - **Timestamp:** $(date +"%F %T")

   ## Day 105: HPC: Memory Throughput & Computation
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Reviewed HPC patterns (PDE solvers, climate modeling) for GPU acceleration.
   - **Timestamp:** $(date +"%F %T")

   ## Day 106: HPC: Precision & Mixed Precision
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used half or custom data types for HPC kernels if feasible.
   - **Timestamp:** $(date +"%F %T")

   ## Day 107: Advanced Debugging Tools (cuda-memcheck, etc.)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Used cuda-memcheck for memory leak/race detection in a bigger scenario.
   - **Timestamp:** $(date +"%F %T")

   ## Day 108: Graphics Interop (OpenGL/DX)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** If relevant, shared buffers between CUDA and graphics APIs.
   - **Timestamp:** $(date +"%F %T")

   ## Day 109: Large-Scale Code, Maintainability
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Organized your code into modules/libraries; considered CMake for builds.
   - **Timestamp:** $(date +"%F %T")

   ## Day 110: HPC Tools & Libraries (MAGMA, etc.)
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Tried MAGMA for advanced linear algebra on GPU.
   - **Timestamp:** $(date +"%F %T")

   ## Day 111: Testing & Validation Strategies
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Implemented unit tests for GPU kernels using CPU reference checks.
   - **Timestamp:** $(date +"%F %T")

   ## Day 112: Progress Checkpoint
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Reflect on HPC/ML techniques, debugging, multi-GPU scaling.
   - **Timestamp:** $(date +"%F %T")

   ## Day 113: Revisiting Key Optimizations
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Identified top 3 bottlenecks in your main code; systematically addressed them.
   - **Timestamp:** $(date +"%F %T")

   ## Day 114: GPU Scheduling & CUcontext Exploration
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Investigated multiple contexts/users sharing GPU resources.
   - **Timestamp:** $(date +"%F %T")

   ## Day 115: Final Performance Tweaks & Fine-Tuning
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Adjusted L1/Shared memory config if your GPU allows; fine-tuned block dimensions.
   - **Timestamp:** $(date +"%F %T")

   ## Day 116: Memory Hierarchy Mastery
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Created a reference diagram of global, shared, local, constant, texture, etc.
   - **Timestamp:** $(date +"%F %T")

   ## Day 117: Detailed Profiling Recap
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Re-profiled older mini-projects; applied new knowledge for more gains.
   - **Timestamp:** $(date +"%F %T")

   ## Day 118: Review of Common Pitfalls
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Made a checklist of frequent issues: out-of-bounds, race conditions, divergence, etc.
   - **Timestamp:** $(date +"%F %T")

   ## Day 119: Prep for Final Capstone
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Checked environment, references, library versions; planned scope carefully.
   - **Timestamp:** $(date +"%F %T")

   ## Day 120: Capstone Project #6
   - **Commit:** $(git rev-parse --short HEAD)
   - **Description:** Final Project: End-to-End HPC or ML Application.
   - **Timestamp:** $(date +"%F %T")

