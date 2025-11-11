
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

__global__ void naive_atten_kernel(
    const float *Q, // [batch_size, seq_len, dim]
    const float *K,
    const float *V,
    float * output, // [batch_size, seq_len, dim]
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

torch::Tensor naive_atten(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    
    int batch_size = Q.size(0);
    int seq_len = Q.size(1);
    int dim = Q.size(2);

    auto output = torch::zeros_like(Q);

    dim3 block(16, 16);
    dim3 grid(
        (dim + block.x - 1) / block.x,
        (seq_len + block.y - 1) / block.y,
        batch_size
    );

    naive_atten_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in attention_naive_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "naive_atten",
        &naive_atten,
        "Naive Attention CUDA"
    );
}