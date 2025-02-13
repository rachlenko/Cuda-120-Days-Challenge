```markdown
# Day 27: Texture & Surface Memory (Intro)

In this lesson, we introduce **Texture Memory** in CUDA, a special read-only memory space optimized for 2D spatial locality and interpolation. Texture memory is particularly useful for image processing tasks such as sampling, filtering, and edge detection. We will cover the basics of texture memory, how to bind a CUDA array to a texture, and how to sample a small 2D texture in a kernel. We will also compare texture fetch performance with global memory fetch, and discuss common pitfalls such as missing texture binding/unbinding steps.

**Key Learning Objectives:**
- Understand the concepts and advantages of texture memory.
- Learn how to allocate a CUDA array and bind it to a texture reference.
- Implement a simple kernel to sample a 2D texture.
- Compare texture memory fetch performance with global memory fetch.
- Identify common debugging pitfalls and best practices.
- Use detailed code examples with extensive inline comments.
- Provide conceptual diagrams to illustrate the workflow.

*References:*
- [CUDA C Programming Guide – Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)
- [CUDA Samples – Texture Memory](https://docs.nvidia.com/cuda/cuda-samples/index.html)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [What is Texture Memory?](#2-what-is-texture-memory)  
3. [Setting Up Texture Memory in CUDA](#3-setting-up-texture-memory-in-cuda)  
4. [Practical Exercise: Sampling a Small 2D Texture](#4-practical-exercise-sampling-a-small-2d-texture)  
    - [a) Kernel Code: Texture Sampling vs. Global Memory Fetch](#a-kernel-code-texture-sampling-vs-global-memory-fetch)  
    - [b) Host Code: Binding and Unbinding Texture Memory](#b-host-code-binding-and-unbinding-texture-memory)  
5. [Common Debugging Pitfalls and Best Practices](#5-common-debugging-pitfalls-and-best-practices)  
6. [Conceptual Diagrams](#6-conceptual-diagrams)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)  

---

## 1. Overview

Texture memory is a **read-only** memory space on the GPU that is optimized for 2D spatial access patterns. It provides benefits such as:
- **Hardware caching** for faster access.
- **Built-in addressing modes** (e.g., clamping, wrapping) useful for image processing.
- **Interpolation support** for filtering operations.

Unlike global memory, texture fetches are optimized for **spatial locality**; when threads access nearby data elements, the texture cache can dramatically improve performance. However, to use texture memory properly, you must correctly bind the data to a texture object and unbind it when finished.

---

## 2. What is Texture Memory?

**Texture Memory** in CUDA is:
- **Read-only** during kernel execution.
- Cached on the GPU for faster access.
- Ideal for accessing image-like data with 2D spatial locality.

It is typically used by:
- **Image processing algorithms** (e.g., filtering, edge detection).
- **3D graphics applications.**

**Important:**  
Texture memory requires proper binding of CUDA arrays (or linear memory) to a texture object before the kernel can sample from it.

---

## 3. Setting Up Texture Memory in CUDA

To use texture memory:
1. **Allocate a CUDA Array:**  
   Use `cudaMallocArray()` to allocate a 2D array on the device.
2. **Copy Data to the CUDA Array:**  
   Use `cudaMemcpy2DToArray()` to copy image data to the CUDA array.
3. **Create a Texture Object:**  
   Set up a `cudaResourceDesc` and `cudaTextureDesc`, then call `cudaCreateTextureObject()`.
4. **Use the Texture Object in a Kernel:**  
   Sample the texture in your kernel using functions like `tex2D()`.
5. **Destroy the Texture Object:**  
   After kernel execution, release the texture object with `cudaDestroyTextureObject()`.

---

## 4. Practical Exercise: Sampling a Small 2D Texture

In this exercise, we implement a simple kernel that compares sampling from a 2D texture with a direct global memory fetch. We will use a small 2D image for demonstration.

### a) Kernel Code: Texture Sampling vs. Global Memory Fetch

```cpp
// textureSampleKernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Declare a texture object globally. Note that in CUDA 5.0 and later,
// we use texture objects rather than texture references.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Kernel that samples data using texture memory and compares it with a global memory fetch.
__global__ void textureVsGlobalKernel(const float *globalData, float *outputTex, float *outputGlobal, int width, int height) {
    // Calculate pixel coordinates.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds.
    if (x < width && y < height) {
        // Calculate normalized texture coordinates.
        // Note: tex2D() expects coordinates in float format.
        float u = x + 0.5f;
        float v = y + 0.5f;
        
        // Fetch the value from the texture.
        float texVal = tex2D(texRef, u, v);
        
        // Compute the index for global memory.
        int idx = y * width + x;
        // Fetch the value directly from global memory.
        float globalVal = globalData[idx];
        
        // Write both values to output arrays for comparison.
        outputTex[idx] = texVal;
        outputGlobal[idx] = globalVal;
    }
}
```

*Detailed Comments:*
- The kernel computes 2D coordinates (x, y) for each pixel.
- It calculates normalized coordinates to sample the texture with `tex2D()`.
- It also reads the same value directly from a global memory array.
- The outputs are stored in separate arrays (`outputTex` and `outputGlobal`) for later comparison.

---

### b) Host Code: Binding and Unbinding Texture Memory

```cpp
// textureSampleHost.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel declaration.
__global__ void textureVsGlobalKernel(const float *globalData, float *outputTex, float *outputGlobal, int width, int height);

// Texture reference is declared globally (for texture objects we can also use cudaTextureObject_t, but here we use the legacy texture reference for simplicity).
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

int main() {
    // Image dimensions.
    int width = 512, height = 512;
    size_t size = width * height * sizeof(float);

    // Allocate host memory for image and output arrays.
    float *h_image = (float*)malloc(size);
    float *h_outputTex = (float*)malloc(size);
    float *h_outputGlobal = (float*)malloc(size);
    if (!h_image || !h_outputTex || !h_outputGlobal) {
        printf("Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize image with random values.
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_image[i] = (float)(rand() % 256) / 255.0f;
    }

    // Allocate CUDA array for the 2D texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Copy image data from host to CUDA array.
    cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Set texture parameters (address mode, filter mode, etc.).
    texRef.addressMode[0] = cudaAddressModeClamp;  // Clamp coordinates
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;         // No filtering
    texRef.normalized = false;                       // Use unnormalized texture coordinates

    // Bind the CUDA array to the texture reference.
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // Allocate device memory for global data and output arrays.
    float *d_image, *d_outputTex, *d_outputGlobal;
    cudaMalloc((void**)&d_image, size);
    cudaMalloc((void**)&d_outputTex, size);
    cudaMalloc((void**)&d_outputGlobal, size);

    // Copy the same image data to device global memory.
    cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel.
    textureVsGlobalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_outputTex, d_outputGlobal, width, height);
    cudaDeviceSynchronize();

    // Copy results from device to host.
    cudaMemcpy(h_outputTex, d_outputTex, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputGlobal, d_outputGlobal, size, cudaMemcpyDeviceToHost);

    // Compare outputs (for demonstration, we print the first 10 elements).
    printf("First 10 values from texture fetch:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_outputTex[i]);
    }
    printf("\n");

    printf("First 10 values from global memory fetch:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_outputGlobal[i]);
    }
    printf("\n");

    // Unbind the texture.
    cudaUnbindTexture(texRef);

    // Free device memory and CUDA array.
    cudaFree(d_image);
    cudaFree(d_outputTex);
    cudaFree(d_outputGlobal);
    cudaFreeArray(cuArray);

    // Free host memory.
    free(h_image);
    free(h_outputTex);
    free(h_outputGlobal);

    return 0;
}
```

*Detailed Comments:*
- **Host Memory Allocation:**  
  Allocate memory for the image and output arrays on the host.
- **CUDA Array Allocation:**  
  Use `cudaMallocArray()` to allocate a 2D CUDA array for the texture.
- **Copy Data to CUDA Array:**  
  Use `cudaMemcpy2DToArray()` to copy the image data into the CUDA array.
- **Texture Parameter Setup:**  
  Set texture address mode, filter mode, and normalization parameters.
- **Binding the Texture:**  
  Bind the CUDA array to the texture reference with `cudaBindTextureToArray()`.
- **Device Memory Allocation for Global Data:**  
  Allocate a separate device array to hold the same image data for comparison.
- **Kernel Launch:**  
  Launch the kernel to sample the texture and fetch data from global memory.
- **Result Comparison:**  
  Copy the outputs back to the host and print them for comparison.
- **Unbinding and Cleanup:**  
  Unbind the texture using `cudaUnbindTexture()`, free device memory, and free host memory.

---

## 6. Common Debugging Pitfalls

| **Pitfall**                               | **Solution**                                             |
|-------------------------------------------|----------------------------------------------------------|
| Missing texture binding                   | Always bind the CUDA array to the texture reference before launching the kernel. |
| Forgetting to unbind the texture          | Unbind texture with `cudaUnbindTexture()` after kernel execution to avoid unintended side effects in subsequent operations. |
| Incorrect texture parameters              | Ensure correct address modes, filter mode, and normalization settings for your use case. |
| Mismatched memory types                   | Do not mix texture fetches with non-texture global memory accesses without proper validation. |
| Not checking for CUDA errors              | Use `cudaGetErrorString()` to log any errors after CUDA API calls. |

---

## 7. Conceptual Diagrams

### Diagram 1: Texture Memory Workflow
```mermaid
flowchart TD
    A[Host: Allocate Image Data (Pageable/Pinned Memory)]
    B[Host: Allocate CUDA Array via cudaMallocArray()]
    C[Host: Copy Image Data to CUDA Array (cudaMemcpy2DToArray)]
    D[Set Texture Parameters (addressMode, filterMode, normalized)]
    E[Bind CUDA Array to Texture Reference (cudaBindTextureToArray)]
    F[Kernel: Sample Data using tex2D()]
    G[Kernel: Also fetch data from Global Memory for Comparison]
    H[Host: Unbind Texture (cudaUnbindTexture)]
    I[Host: Copy Kernel Output from Device to Host]
    J[Host: Compare and Verify Results]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> I
    I --> J
    J --> H
```

*Explanation:*  
- The diagram illustrates the workflow from allocating image data, copying it to a CUDA array, binding it to a texture, sampling it in the kernel, and finally unbinding and copying results back to the host.

### Diagram 2: Kernel Execution Flow for Texture Sampling

```mermaid
flowchart TD
    A[Kernel Launch]
    B[Each Thread Computes (x, y) Coordinates]
    C[Compute Normalized Texture Coordinates]
    D[Fetch Value using tex2D(texRef, u, v)]
    E[Fetch Value from Global Memory]
    F[Store Both Values in Output Arrays]
    
    A --> B
    B --> C
    C --> D
    C --> E
    D & E --> F
```

*Explanation:*  
- This diagram details the steps each thread takes within the kernel.
- Threads compute their coordinates, fetch values from both texture memory and global memory, and store the results for later comparison.

---

## 8. References & Further Reading

1. **CUDA C Programming Guide – Texture Memory**  
   [CUDA Texture Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)  
   Comprehensive guide to texture memory in CUDA.
2. **CUDA C Best Practices Guide – Texture Memory**  
   [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#texture-memory)  
   Best practices for using texture memory effectively.
3. **NVIDIA CUDA Samples – Texture Memory**  
   [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)  
   Example codes provided by NVIDIA for texture memory usage.
4. **NVIDIA Developer Blog – Texture Memory**  
   [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)  
   Articles discussing optimization and usage of texture memory.
5. **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu**  
   A comprehensive resource covering CUDA memory hierarchies including texture memory.

---

## 9. Conclusion

In Day 27, you have learned:
- **The basics of texture memory** and its benefits for 2D spatial data access.
- **How to allocate a CUDA array and bind it to a texture reference.**
- **How to sample a texture in a CUDA kernel** using `tex2D()`, and compare it to global memory fetch.
- **The importance of correct texture binding and unbinding** to avoid bugs.
- **Detailed code examples** with extensive comments and conceptual diagrams to reinforce understanding.

---

## 10. Next Steps

- **Experiment:**  
  Extend this project to implement advanced image processing algorithms (e.g., Sobel edge detection, Gaussian blur) using texture memory.
- **Profile:**  
  Use NVIDIA NSight Compute to compare performance differences between texture fetches and global memory fetches.
- **Optimize:**  
  Experiment with different texture parameters (filter mode, address mode, normalized coordinates) to find the optimal configuration for your application.
- **Integrate:**  
  Combine texture memory with other optimization techniques (such as shared memory and asynchronous transfers) in larger projects.

Happy CUDA coding, and continue to harness the power of texture memory for high-performance image processing!
```

```markdown
# Day 27: Capstone Project – Advanced Image Processing with Texture Memory

In this capstone project, we extend our understanding of texture memory by implementing a complete **Sobel Edge Detection** pipeline using CUDA texture memory. In this project, we will:
- Load a 2D image into a CUDA array.
- Bind the CUDA array to a texture reference.
- Launch a CUDA kernel that samples from the texture using `tex2D()`, applies the Sobel operator in both the X and Y directions, and computes the gradient magnitude.
- Compare the performance and accuracy against a global memory-based implementation.
- Unbind the texture and clean up resources.

This project demonstrates the benefits of texture memory for image processing, particularly in applications such as edge detection where spatial locality and hardware interpolation can be leveraged for improved performance.

---

## Table of Contents

1. [Overview](#overview)  
2. [What is Texture Memory?](#what-is-texture-memory)  
3. [Sobel Edge Detection Algorithm](#sobel-edge-detection-algorithm)  
4. [Implementation Steps](#implementation-steps)  
    - [a) Allocate and Copy Image Data to a CUDA Array](#a-allocate-and-copy-image-data-to-a-cuda-array)  
    - [b) Bind the CUDA Array to a Texture Reference](#b-bind-the-cuda-array-to-a-texture-reference)  
    - [c) Sobel Edge Detection Kernel Using Texture Memory](#c-sobel-edge-detection-kernel-using-texture-memory)  
    - [d) Host Code for Launching the Kernel and Unbinding](#d-host-code-for-launching-the-kernel-and-unbinding)  
5. [Debugging Pitfalls](#debugging-pitfalls)  
6. [Conceptual Diagrams](#conceptual-diagrams)  
7. [References & Further Reading](#references--further-reading)  
8. [Conclusion](#conclusion)  
9. [Next Steps](#next-steps)  

---

## 1. Overview

Texture memory is a read-only memory optimized for 2D spatial locality and is highly effective for image processing. In this project, we implement the **Sobel Edge Detection** algorithm to detect edges in an image using texture memory. The pipeline includes:
- **Memory allocation:** Using `cudaMallocArray()` to create a CUDA array for the image.
- **Data transfer:** Copying image data to the CUDA array using `cudaMemcpy2DToArray()`.
- **Texture binding:** Binding the CUDA array to a texture reference and setting texture parameters.
- **Kernel execution:** Sampling the image using `tex2D()` in the kernel, applying Sobel filters, and computing the gradient magnitude.
- **Resource cleanup:** Unbinding the texture and freeing all allocated memory.

---

## 2. What is Texture Memory?

Texture memory is a specialized region of memory that:
- Is **read-only** during kernel execution.
- Is **cached** on-chip, leading to faster access when spatial locality is present.
- Supports **built-in addressing modes** (e.g., clamp, wrap) and filtering (e.g., linear interpolation).

This makes it ideal for tasks like image processing where nearby pixels are accessed together.

---

## 3. Sobel Edge Detection Algorithm

The Sobel operator is used to calculate the gradient of image intensity at each pixel, emphasizing regions with high spatial frequency that correspond to edges.

**Sobel Kernels:**

- **Sobel X:**
  \[
  G_x = \begin{bmatrix}
  -1 & 0 & 1 \\
  -2 & 0 & 2 \\
  -1 & 0 & 1
  \end{bmatrix}
  \]

- **Sobel Y:**
  \[
  G_y = \begin{bmatrix}
  -1 & -2 & -1 \\
  0 &  0 &  0 \\
  1 &  2 &  1
  \end{bmatrix}
  \]

**Gradient Magnitude Calculation:**
\[
G = \sqrt{G_x^2 + G_y^2}
\]

In our CUDA kernel, we will sample the image from texture memory and apply these filters to compute the gradient magnitude.

---

## 4. Implementation Steps

### a) Allocate and Copy Image Data to a CUDA Array

We first allocate a CUDA array that will hold our 2D image data and then copy the image from host to device.

```cpp
// Host code snippet for allocating and copying image data.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int width = 512, height = 512;
size_t imageSize = width * height * sizeof(float);

// Allocate host memory for the image.
float *h_image = (float*)malloc(imageSize);
if (!h_image) {
    printf("Host memory allocation failed\n");
    exit(EXIT_FAILURE);
}

// Initialize the image with random grayscale values.
srand(time(NULL));
for (int i = 0; i < width * height; i++) {
    h_image[i] = (float)(rand() % 256) / 255.0f; // values between 0 and 1
}

// Create a channel descriptor for a single float.
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

// Allocate a CUDA array for the 2D texture.
cudaArray_t cuArray;
cudaMallocArray(&cuArray, &channelDesc, width, height);

// Copy the image data from host to the CUDA array.
cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
```

*Comments:*
- The image is initialized with random grayscale values.
- A CUDA array is allocated using `cudaMallocArray()` with a channel descriptor for a float.
- `cudaMemcpy2DToArray()` is used to copy the 2D image data.

---

### b) Bind the CUDA Array to a Texture Reference

Next, we bind the CUDA array to a texture reference and set texture parameters.

```cpp
// Texture binding and configuration.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

texRef.addressMode[0] = cudaAddressModeClamp;   // Clamp x coordinates
texRef.addressMode[1] = cudaAddressModeClamp;   // Clamp y coordinates
texRef.filterMode = cudaFilterModePoint;        // No interpolation (point sampling)
texRef.normalized = false;                      // Use unnormalized coordinates

// Bind the CUDA array to the texture reference.
cudaBindTextureToArray(texRef, cuArray, channelDesc);
```

*Comments:*
- The texture reference `texRef` is declared as a global variable.
- Texture parameters are set: clamping for addressing, point sampling for filtering, and unnormalized coordinates.
- The CUDA array is bound to the texture reference with `cudaBindTextureToArray()`.

---

### c) Sobel Edge Detection Kernel Using Texture Memory

The kernel will sample from the texture using `tex2D()`, apply the Sobel operator in both X and Y directions, compute the gradient magnitude, and store the result in an output array.

```cpp
// sobelEdgeKernel.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Declare the texture reference for reading the image.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Sobel Edge Detection Kernel.
// This kernel computes the gradient magnitude using the Sobel operator.
__global__ void sobelEdgeKernel(float *output, int width, int height) {
    // Calculate the pixel coordinates.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process valid pixels (avoid boundaries)
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sample the image from texture memory using tex2D.
        // The texture coordinates are in pixel units since normalized is false.
        // Fetch neighboring pixel values for Sobel operator.
        float Gx = -tex2D(texRef, x - 1, y - 1) - 2.0f * tex2D(texRef, x - 1, y) - tex2D(texRef, x - 1, y + 1)
                   + tex2D(texRef, x + 1, y - 1) + 2.0f * tex2D(texRef, x + 1, y) + tex2D(texRef, x + 1, y + 1);
                   
        float Gy = -tex2D(texRef, x - 1, y - 1) - 2.0f * tex2D(texRef, x, y - 1) - tex2D(texRef, x + 1, y - 1)
                   + tex2D(texRef, x - 1, y + 1) + 2.0f * tex2D(texRef, x, y + 1) + tex2D(texRef, x + 1, y + 1);
        
        // Compute the gradient magnitude.
        float gradMag = sqrtf(Gx * Gx + Gy * Gy);
        
        // Write the result to the output array.
        output[y * width + x] = gradMag;
    }
}
```

*Comments:*
- Each thread computes its (x, y) coordinate.
- Boundary conditions are applied (processing only pixels where neighbors exist).
- The kernel uses `tex2D()` to fetch the 3x3 neighborhood pixel values.
- The Sobel operators for the X and Y directions are applied.
- The gradient magnitude is calculated as `sqrt(Gx^2 + Gy^2)`.
- The computed edge strength is stored in the output array.

---

### d) Host Code with Detailed Error Checking and Texture Binding

```cpp
// sobelEdgeDetection.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel declaration.
__global__ void sobelEdgeKernel(float *output, int width, int height);

// Declare the texture reference globally.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

#define CUDA_CHECK(call) {                                      \
    cudaError_t err = call;                                     \
    if(err != cudaSuccess) {                                   \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

int main() {
    // Set image dimensions.
    int width = 512, height = 512;
    size_t imgSize = width * height * sizeof(float);

    // Allocate host memory for the image and output.
    float *h_image = (float*)malloc(imgSize);
    float *h_output = (float*)malloc(imgSize);
    if (!h_image || !h_output) {
        printf("Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host image with random values.
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_image[i] = (float)(rand() % 256) / 255.0f;
    }

    // Allocate a CUDA array for the 2D texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Copy image data from host to the CUDA array.
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));

    // Set texture parameters.
    texRef.addressMode[0] = cudaAddressModeClamp;  // Clamp x coordinate
    texRef.addressMode[1] = cudaAddressModeClamp;  // Clamp y coordinate
    texRef.filterMode = cudaFilterModePoint;       // Point sampling (no interpolation)
    texRef.normalized = false;                     // Use unnormalized coordinates

    // Bind the CUDA array to the texture reference.
    CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    // Allocate device memory for output.
    float *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, imgSize));

    // Define kernel launch parameters.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the Sobel edge detection kernel.
    sobelEdgeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from device to host.
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));

    // (Optional) Print first 10 values for verification.
    printf("First 10 edge values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Unbind the texture.
    CUDA_CHECK(cudaUnbindTexture(texRef));

    // Free resources.
    CUDA_CHECK(cudaFreeArray(cuArray));
    CUDA_CHECK(cudaFree(d_output));
    free(h_image);
    free(h_output);

    return 0;
}
```

*Detailed Comments:*
- **Host Memory Allocation and Initialization:**  
  Host memory is allocated for the image and output, and the image is initialized with random grayscale values.
- **CUDA Array Allocation:**  
  A CUDA array is allocated to store the image. This array is used as the data source for the texture.
- **Data Transfer:**  
  `cudaMemcpy2DToArray()` copies the image data from host memory to the CUDA array.
- **Texture Binding:**  
  Texture parameters (address mode, filter mode, normalization) are set. The CUDA array is bound to the texture reference using `cudaBindTextureToArray()`.
- **Kernel Launch:**  
  The Sobel edge detection kernel is launched with a 2D grid configuration.
- **Result Transfer:**  
  After kernel execution, the output is copied from device to host.
- **Resource Cleanup:**  
  The texture is unbound with `cudaUnbindTexture()`, and all allocated resources are freed.

---

## 6. Common Debugging Pitfalls

| **Pitfall**                                      | **Solution**                                                 |
|--------------------------------------------------|--------------------------------------------------------------|
| Missing texture binding/unbinding                | Always bind the CUDA array to the texture reference before kernel launch and unbind after execution using `cudaUnbindTexture()`. |
| Incorrect texture parameter settings             | Ensure address modes, filter mode, and normalized settings are configured correctly. |
| Using normalized coordinates when not intended   | Set `normalized = false` if coordinates are in pixel units.  |
| Memory copy errors from host to CUDA array       | Verify dimensions and pitch in `cudaMemcpy2DToArray()`.      |
| Failing to check for CUDA errors                 | Use error-checking macros (e.g., `CUDA_CHECK`) after each CUDA call. |

---

## 7. Conceptual Diagrams

### Diagram 1: Texture Memory Workflow for Image Convolution

```mermaid
flowchart TD
    A[Host: Load Image Data]
    B[Allocate CUDA Array using cudaMallocArray()]
    C[Copy Image Data to CUDA Array using cudaMemcpy2DToArray()]
    D[Set Texture Parameters (address, filter mode, normalization)]
    E[Bind CUDA Array to Texture Reference (cudaBindTextureToArray)]
    F[Kernel: Sample Texture using tex2D()]
    G[Kernel: Apply Sobel Operator (compute Gx, Gy, gradient magnitude)]
    H[Write Output to Device Memory]
    I[Host: Copy Output from Device to Host]
    J[Unbind Texture (cudaUnbindTexture)]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
```

*Explanation:*  
- The diagram illustrates the flow of data from host memory to the CUDA array and then binding to texture memory.  
- The kernel samples the texture and computes the Sobel edge detection, with results copied back to the host and finally unbinding the texture.

### Diagram 2: Kernel Execution Flow for Sobel Edge Detection

```mermaid
flowchart TD
    A[Kernel Launch]
    B[Each thread calculates its (x, y) pixel coordinates]
    C[Compute texture coordinates (u, v)]
    D[Sample texture using tex2D(texRef, u, v)]
    E[Fetch neighboring pixels for Sobel filter]
    F[Compute Gx and Gy using Sobel kernels]
    G[Calculate gradient magnitude G = sqrt(Gx^2 + Gy^2)]
    H[Store result in output array]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

*Explanation:*  
- Each thread computes its pixel position and samples texture data.
- The Sobel operator is applied by fetching a 3x3 neighborhood, computing gradients, and then calculating the final edge magnitude.

---

## 8. References & Further Reading

1. **CUDA C Programming Guide – Texture Memory**  
   [CUDA Texture Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)
2. **CUDA C Best Practices Guide – Texture Memory**  
   [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#texture-memory)
3. **NVIDIA CUDA Samples – Texture Memory**  
   [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)
4. **NVIDIA Developer Blog – Texture Memory**  
   [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
5. **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu**  
6. **CUDA C Programming Guide – Asynchronous Transfers & Memory Management**  
   [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory)

---

## 9. Conclusion

In Day 27, we have:
- Introduced **texture memory** for high-performance 2D data access.
- Detailed how to **bind a CUDA array to a texture reference**.
- Implemented a **Sobel edge detection kernel** that samples from texture memory.
- Provided **extensive code examples with detailed inline comments**.
- Illustrated the workflow with **conceptual diagrams**.
- Discussed **common pitfalls** such as missing binding or incorrect texture parameters.

---

## 10. Next Steps

- **Extend the Project:**  
  Implement additional image processing filters (e.g., Gaussian blur, sharpening) using texture memory.
- **Profile and Optimize:**  
  Use NVIDIA NSight Compute to compare texture fetch performance with global memory fetches.
- **Integrate with Other Techniques:**  
  Combine texture memory with shared memory and asynchronous transfers in larger applications, such as real-time video processing or deep learning inference.
- **Experiment with Texture Filtering:**  
  Try different filter modes (e.g., linear interpolation) and address modes (wrap, clamp) to understand their effect on image processing quality and performance.

Happy CUDA coding, and continue to push the boundaries of GPU-accelerated image processing!
```
