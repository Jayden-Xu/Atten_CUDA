import torch
from torch.utils.cpp_extension import load
import os
import tqdm

from pytorch_reference import sparse_attention_pytorch 

cuda_source = os.path.join(
    os.path.dirname(__file__), 
    '../kernels/block_sparse_atten.cu'
)

block_sparse_atten = load(
    name = "block_sparse_atten",
    sources = [cuda_source],
    extra_cflags = ['-O3'],
    verbose = True
)

TILE_SIZE = 32

def create_block_layout(seq_len, pattern='dense'):
    """
    Creates the [num_q_tiles, num_k_tiles] INT32 layout 
    for the CUDA kernel.
    """

    num_q_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    num_k_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    
    if pattern == 'dense':
        return torch.ones(num_q_tiles, num_k_tiles, dtype=torch.int32, device='cuda')
    
    if pattern == 'sliding_window':
        window_size_in_blocks = num_q_tiles // 4
        q_idx = torch.arange(num_q_tiles, device='cuda')
        k_idx = torch.arange(num_k_tiles, device='cuda')
        mask = (q_idx[:, None] - k_idx[None, :]).abs() <= window_size_in_blocks
        return mask.to(torch.int32)

    if pattern == 'imbalanced':
        layout = torch.zeros(num_q_tiles, num_k_tiles, dtype=torch.int32, device='cuda')
        layout[0::2, :] = 1
        layout[1::2, 0] = 1
        return layout

    raise ValueError("Unknown pattern")


def un_tile_mask(block_layout, seq_len):
    """
    Converts a [num_q, num_k] block layout into a 
    [seq_len, seq_len] boolean mask for the PyTorch baseline.
    """
    
    token_mask = torch.kron(
        block_layout.to(torch.bool), 
        torch.ones(TILE_SIZE, TILE_SIZE, device='cuda', dtype=torch.bool)
    )

    return token_mask[:seq_len, :seq_len]

def test_block_sparse_atten():
    print("Testing General Block-Sparse Attention correctness...")
    batch_size, seq_len, dim = 4, 1024, 64 

    torch.manual_seed(0)
    Q = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)

    block_layout = create_block_layout(seq_len, pattern='imbalanced')

    token_mask = un_tile_mask(block_layout, seq_len)

    true_output = sparse_attention_pytorch(Q, K, V, token_mask)
    result = block_sparse_atten.block_sparse_atten(Q, K, V, block_layout)

    if torch.allclose(result, true_output, atol=1e-4, rtol=1e-5):
        print("Success: Kernel output matches PyTorch block-sparse reference.")
    else:
        error = (result - true_output).abs().max().item()
        print(f"Failure: Max error = {error}")


def benchmark_load_imbalance():
    print("\nRunning Load Imbalance Benchmark...")
    batch_size, seq_len, dim = 16, 4096, 64
    num_warmup, num_iterations = 10, 100

    Q = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)

    dense_layout = create_block_layout(seq_len, 'dense')
    imbalanced_layout = create_block_layout(seq_len, 'imbalanced')
    
    # calculate sparsity
    density = imbalanced_layout.sum() / dense_layout.sum()
    print(f"Imbalanced mask density: {density.item() * 100:.2f}%")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Benchmarking CUDA (Dense)...")
    for _ in tqdm.trange(num_warmup, desc="CUDA Dense Warmup"):
        _ = block_sparse_atten.block_sparse_atten(Q, K, V, dense_layout)
    torch.cuda.synchronize()
    start_event.record()
    for _ in tqdm.trange(num_iterations, desc="CUDA Dense Bench"):
        _ = block_sparse_atten.block_sparse_atten(Q, K, V, dense_layout)
    end_event.record()
    torch.cuda.synchronize()
    avg_dense_ms = start_event.elapsed_time(end_event) / num_iterations

    print(f"Benchmarking CUDA (Imbalanced)...")
    for _ in tqdm.trange(num_warmup, desc="CUDA Imbalanced Warmup"):
        _ = block_sparse_atten.block_sparse_atten(Q, K, V, imbalanced_layout)
    torch.cuda.synchronize()
    start_event.record()
    for _ in tqdm.trange(num_iterations, desc="CUDA Imbalanced Bench"):
        _ = block_sparse_atten.block_sparse_atten(Q, K, V, imbalanced_layout)
    end_event.record()
    torch.cuda.synchronize()
    avg_imbalanced_ms = start_event.elapsed_time(end_event) / num_iterations

    print(f"\n--- Load Imbalance Results ---")
    print(f"Setup: Batch={batch_size}, SeqLen={seq_len}, Dim={dim}")
    print(f"Average CUDA (Dense) time: {avg_dense_ms:.3f} ms")
    print(f"Average CUDA (Imbalanced) time: {avg_imbalanced_ms:.3f} ms")
    
    speedup = avg_dense_ms / avg_imbalanced_ms
    print(f"\nSpeedup: {speedup:.3f}x")
    print(f"Density: {density.item():.3f}")
    
    print(f"Achieved {speedup:.3f}x speedup with {density.item():.3f} density.")
    if speedup < (1/density.item()) * 0.8:
        print(f"This is << ideal ({1/density.item():.3f}x). This confirms LOAD IMBALANCE.")
    else:
        print(f"This is close to ideal ({1/density.item():.3f}x).")


if __name__ == '__main__':
    test_block_sparse_atten()
    benchmark_load_imbalance()