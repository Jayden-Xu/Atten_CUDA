
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

__global__ void vector_add_kerenl(float *a, float *b, float *c, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_cpu(float *a, float *b, float *c, int n) {

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void verify_result(float *c, float *d_c, int n) {

    int eps = 1e-4f;
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - d_c[i]) > eps) {
            printf("result verification failed at element %d!\n", i);
            break;
        }
    }
    printf("result verification passed!\n");
}

int main() {

    int n = 10000;
    size_t size = n * sizeof(float);

    // allocate host mem
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);
    float *c_cpu = (float *)malloc(size);

    // allocate device mem
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // initialize host mem
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // copy mem from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    vector_add_kerenl<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %f ms\n", milliseconds);


    // copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    // verify
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vector_add_cpu(a, b, c_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    float cpu_milliseconds = duration.count() / 1000.0f;

    printf("CPU time (chrono): %f ms\n", cpu_milliseconds);

    verify_result(c, c_cpu, n);

    // free device mem
    cudaFree(d_a);
    cudaFree(d_b);n
    cudaFree(d_c);

    // free host mem
    free(a);
    free(b);
    free(c);
    free(c_cpu);

}