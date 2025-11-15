
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
using namespace std;

__global__ void naive_atten_kernel(
    const float* __restrict__ Q, // [batch_size, seq_len, dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output, // [batch_size, seq_len, dim]
    int batch_size,
    int seq_len,
    int dim
) {
    
    int b = blockIdx.z; // batch
    int i = blockIdx.y * blockDim.y + threadIdx.y; // seq_len
    int d = blockIdx.x * blockDim.x + threadIdx.x; // dim

    if (b >= batch_size || i >= seq_len || d >= dim) {
        return ;
    }

    int d_k = sqrtf((float)dim);

    float sum_exp = 0.0f;
    float max_score = -INFINITY;

    // find the max for numerical stability
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;

        for (int k = 0; k < dim; k++) {
            int q_idx = b * seq_len * dim + i * dim + k;
            int k_idx = b * seq_len * dim + j * dim + k;
            score += Q[q_idx] * K[k_idx];
        }

        score /= d_k;

        if (score > max_score) {
            max_score = score;
        }
        
    }

    // 2nd pass to calculate softmax
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;

        for (int k = 0; k < dim; k++) {
            int q_idx = b * seq_len * dim + i * dim + k;
            int k_idx = b * seq_len * dim + j * dim + k;
            score += Q[q_idx] * K[k_idx];
        }

        score /= d_k;
        sum_exp += expf(score - max_score);
    }

    float result = 0.0f;
    // 3rd pass to compute weighted sum
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;

        for (int k = 0; k < dim; k++) {
            int q_idx = b * seq_len * dim + i * dim + k;
            int k_idx = b * seq_len * dim + j * dim + k;
            score += Q[q_idx] * K[k_idx];
        }

        score /= d_k;
        float atten_weight = expf(score - max_score) / sum_exp;
        
        int v_idx = b * seq_len * dim + j * dim + d;
        result += atten_weight * V[v_idx];

    }

    int out_idx = b * seq_len * dim + i * dim + d;
    output[out_idx] = result;
}

int main() {
    int batch_size = 16;
    int seq_len = 1024;
    int dim = 64;

    int size = batch_size * seq_len * dim;
    const size_t bytes = size * sizeof(float);

    float* Q = (float*)malloc(bytes);
    float* K = (float*)malloc(bytes);
    float* V = (float*)malloc(bytes);
    float* output = (float*)malloc(bytes);

    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_output, bytes);

    // initialize host mem
    for (int i = 0; i < size; i++) {
        Q[i] = 1.0f;
        K[i] = 2.0f;
        V[i] = 3.0f;
    }

    // copy mem from host to device
    cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    naive_atten_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_Q,
        d_K,
        d_V,
        d_output,
        batch_size,
        seq_len,
        dim
    );

    // copy mem from device to host
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // free host mem
    free(Q);
    free(K);
    free(V);
    free(output);

    return 0;
}