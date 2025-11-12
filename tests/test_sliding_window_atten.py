
import torch
from torch.utils.cpp_extension import load
import os

from pytorch_reference import sparse_attention_pytorch

cuda_source = os.path.join(
    os.path.dirname(__file__), 
    '../kernels/sliding_window_atten.cu'
)

sliding_window_atten = load(
    name = "sliding_window_atten",
    sources = [cuda_source],
    extra_cflags = ['-O3'],
    verbose = True
)

def sliding_window_mask(seq_len, window_size_in_blocks):

    TILE_SIZE = 32
    num_q_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    num_k_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    
    q_indices = torch.arange(num_q_tiles, device = 'cuda')
    k_indices = torch.arange(num_k_tiles, device = 'cuda')
    
    block_mask = (q_indices[:, None] - k_indices[None, :]).abs() <= window_size_in_blocks
    
    token_mask = torch.kron(block_mask, 
                            torch.ones(TILE_SIZE, TILE_SIZE, device = 'cuda', dtype=torch.bool))
    
    return token_mask[:seq_len, :seq_len]


def test_sliding_window_atten():
    print("Testing Sliding Window Flash Attention correctness...")
    batch_size, seq_len, dim = 4, 1024, 64 
    window_size_in_blocks = 8 # blocks (8 * 32 = 256 tokens)

    torch.manual_seed(0)
    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)

    mask = sliding_window_mask(seq_len, window_size_in_blocks)
    true_output = sparse_attention_pytorch(Q, K, V, mask)
    
    result = sliding_window_atten.sliding_window_atten(Q, K, V, window_size_in_blocks)

    error = (result - true_output).abs().max().item()

    if error > 1e-4:
        print(f"Error: {error}")
    else:
        print("Success")


def benchmark_sliding_window_atten():

    batch_size, seq_len, dim = 16, 4096, 64
    window_size_in_blocks = 32 # blocks (64 * 32 = 2048 tokens)
    num_warmup, num_iterations = 10, 100

    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    for _ in range(num_warmup):
        _ = sliding_window_atten.sliding_window_atten(Q, K, V, window_size_in_blocks)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = sliding_window_atten.sliding_window_atten(Q, K, V, window_size_in_blocks)
    end_event.record()
    torch.cuda.synchronize()

    avg_sw_time_ms = start_event.elapsed_time(end_event) / num_iterations
    
    mask = sliding_window_mask(seq_len, window_size_in_blocks)
    for _ in range(num_warmup):
        _ = sparse_attention_pytorch(Q, K, V, mask)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = sparse_attention_pytorch(Q, K, V, mask)
    end_event.record()
    torch.cuda.synchronize()

    avg_pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations

    print(f"Setup: Batch={batch_size}, SeqLen={seq_len}, Dim={dim}")
    print(f"Average Sliding Window time: {avg_sw_time_ms:.3f} ms")
    print(f"Average PyTorch (sparse) time: {avg_pytorch_time_ms:.3f} ms")

if __name__ == '__main__':
    test_sliding_window_atten()
    benchmark_sliding_window_atten()