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
