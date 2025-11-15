// The block sparse attention is implemented based on flash attention

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

using Tensor = torch::Tensor;

#define TILE_SIZE 32 
#define MAX_DIM 64

__global__ void block_sparse_atten_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l_global, // rowsum
    float* __restrict__ m_global, // rowmax
    const int batch_size,
    const int N, // seq_len
    const int d, // head_dim
    const int total_kv_blocks,
    const int* __restrict__ block_mask_layout // [num_q_tiles, total_kv_blocks]
) {

    extern __shared__ float sram[];

    float* K_tile = sram;
    float* V_tile = K_tile + TILE_SIZE * d;
    float* Q_tile = V_tile + TILE_SIZE * d;

    float* S_tile = Q_tile + TILE_SIZE * d;
    float* O_tile = S_tile + TILE_SIZE * TILE_SIZE;

    float* l_tile = O_tile + TILE_SIZE * d;
    float* m_tile = l_tile + TILE_SIZE;

    int tx = threadIdx.x;
    int bx = blockIdx.y;  // batch index
    int by = blockIdx.x;  // Q-block index

    // each block processes one Q-block
    int q_start = by * TILE_SIZE;
    int q_len = min(TILE_SIZE, N - q_start);

    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = tx; c < d; c += TILE_SIZE) {
            O_tile[r * d + c] = 0.0f;
        }
    }
    if (tx < TILE_SIZE) {
        l_tile[tx] = 0.0f;
        m_tile[tx] = -INFINITY;
    }
    __syncthreads();

    // load Q_tile
    for (int r = 0; r < q_len; ++r) {
        for (int c = tx; c < d; c += TILE_SIZE) {
            // Global memory index: batch_offset + row_offset + col_offset
            int q_idx = bx * N * d + (q_start + r) * d + c;
            Q_tile[r * d + c] = Q[q_idx];
        }
    }
    __syncthreads();

    // loop over KV blocks
    for (int kv_idx = 0; kv_idx < total_kv_blocks; ++kv_idx) {
        // check the mask layout
        int mask_idx = by * total_kv_blocks + kv_idx;
        if (block_mask_layout[mask_idx] == 0) continue;

        int k_start = kv_idx * TILE_SIZE;
        int k_len = min(TILE_SIZE, N - k_start);

        // load KV tiles
        for (int r = 0; r < k_len; ++r) {
            for (int c = tx; c < d; c += TILE_SIZE) {
                int kv_idx_global = bx * N * d + (k_start + r) * d + c;
                K_tile[r * d + c] = K[kv_idx_global];
                V_tile[r * d + c] = V[kv_idx_global];
            }
        }
        __syncthreads();

        // S_tile[i][j] = Q[i] dot K[j]
        if (tx < q_len) {
            float scale = 1.0f / sqrtf((float)d);
            for (int j = 0; j < k_len; ++j) {
                float score = 0.0f;
                for (int k = 0; k < d; ++k) {
                    score += Q_tile[tx * d + k] * K_tile[j * d + k];
                }
                S_tile[tx * TILE_SIZE + j] = score * scale;
            }
        }
        __syncthreads();

        // online softmax
        if (tx < q_len) {
            // max in the K-tile
            float m_prev = m_tile[tx];
            float m_curr = -INFINITY;
            for (int j = 0; j < k_len; ++j) {
                m_curr = fmaxf(m_curr, S_tile[tx * TILE_SIZE + j]);
            }

            // update global max
            float m_new = fmaxf(m_prev, m_curr);
            m_tile[tx] = m_new;

            float l_prev = l_tile[tx];
            float l_curr = 0.0f;
            for (int j = 0; j < k_len; ++j) {
                float p_val = expf(S_tile[tx * TILE_SIZE + j] - m_new);
                S_tile[tx * TILE_SIZE + j] = p_val;
                l_curr += p_val;
            }
            
            // l_new = e^(m_prev - m_new) * l_prev + e^(m_curr - m_new) * l_curr
            float exp_scale = expf(m_prev - m_new);
            l_tile[tx] = l_prev * exp_scale + l_curr;

            // O_i = O_i * exp_scale + P_i * V_tile
            for (int c = 0; c < d; ++c) {
                float pv_sum = 0.0f;
                for (int j = 0; j < k_len; ++j) {
                    pv_sum += S_tile[tx * TILE_SIZE + j] * V_tile[j * d + c];
                }
                O_tile[tx * d + c] = O_tile[tx * d + c] * exp_scale + pv_sum;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < q_len; ++r) {
        // if the entire row is masked, 0
        float inv_l = (l_tile[r] > 0.0f) ? 1.0f / l_tile[r] : 0.0f;
        for (int c = tx; c < d; c += TILE_SIZE) {
            int o_idx = bx * N * d + (q_start + r) * d + c;
            O[o_idx] = O_tile[r * d + c] * inv_l;
        }
    }
    
    if (tx < q_len) {
        int l_idx = bx * N + q_start + tx;
        l_global[l_idx] = l_tile[tx];
        m_global[l_idx] = m_tile[tx];
    }
}

Tensor block_sparse_atten(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& block_mask
) {
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (batch, seq, dim)");
    TORCH_CHECK(K.dim() == 3, "K must be 3D (batch, seq, dim)");
    TORCH_CHECK(V.dim() == 3, "V must be 3D (batch, seq, dim)");
    TORCH_CHECK(block_mask.dim() == 2, "block_mask must be 2D (num_q_tiles, total_kv_blocks)");

    int batch_size = Q.size(0);
    int N = Q.size(1);
    int d = Q.size(2);

    TORCH_CHECK(d <= MAX_DIM, "Head dimension must be <= ", MAX_DIM);
    TORCH_CHECK(Q.size(2) == d && K.size(2) == d && V.size(2) == d, "All tensors must have same head dimension");

    auto opts = Q.options();
    auto O = torch::empty_like(Q);
    auto l_global = torch::empty({batch_size, N}, opts.dtype(torch::kFloat32));
    auto m_global = torch::empty({batch_size, N}, opts.dtype(torch::kFloat32));

    dim3 block(32); 
    
    // Grid: y is batch, x is Q-block
    int num_q_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(num_q_tiles, batch_size);

    int total_kv_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;

    TORCH_CHECK(block_mask.size(0) == num_q_tiles, "block_mask dim 0 must be num_q_tiles");
    TORCH_CHECK(block_mask.size(1) == total_kv_blocks, "block_mask dim 1 must be total_kv_blocks");

    size_t sram_size = (3 * TILE_SIZE * d + TILE_SIZE * TILE_SIZE + TILE_SIZE * d + 2 * TILE_SIZE) * sizeof(float);

    block_sparse_atten_kernel<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        l_global.data_ptr<float>(),
        m_global.data_ptr<float>(),
        batch_size,
        N,
        d,
        total_kv_blocks,
        block_mask.data_ptr<int>()
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_atten", &block_sparse_atten, "General Block Sparse Flash Attention CUDA");
}