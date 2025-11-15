import torch
from torch.utils.cpp_extension import load
import os

naive_atten = load(
    name = "naive_atten",
    sources = [os.path.join(
        os.path.dirname(__file__), 
        '../kernels/naive_atten.cu'
    )],
    extra_cflags = ['-O3'],
    verbose = True
)

flash_atten = load(
    name = "flash_atten",
    sources = [os.path.join(
        os.path.dirname(__file__), 
        '../kernels/flash_atten.cu'
    )],
    extra_cflags = ['-O3'],
    verbose = True
)

sliding_window_atten = load(
    name = "sliding_window_atten",
    sources = [os.path.join(
        os.path.dirname(__file__), 
        '../kernels/sliding_window_atten.cu'
    )],
    extra_cflags = ['-O3'],
    verbose = True
)

def benchmark():
    batch_size, seq_len, dim = 16, 4096, 64
    window_size_in_blocks = 32 # blocks (64 * 32 = 2048 tokens)
    num_warmup, num_iterations = 10, 100

    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    '''
    # sliding window atten
    for _ in range(num_warmup):
        _ = sliding_window_atten.sliding_window_atten(Q, K, V, window_size_in_blocks)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = sliding_window_atten.sliding_window_atten(Q, K, V, window_size_in_blocks)
    end_event.record()
    torch.cuda.synchronize()

    avg_sw_time_ms = start_event.elapsed_time(end_event) / num_iterations
    '''

    # flash atten
    for _ in range(num_warmup):
        _ = flash_atten.flash_atten(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = flash_atten.flash_atten(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()

    avg_flash_time_ms = start_event.elapsed_time(end_event) / num_iterations
    
    # naive atten
    for _ in range(num_warmup):
        _ = naive_atten.naive_atten(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = naive_atten.naive_atten(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()

    avg_naive_time_ms = start_event.elapsed_time(end_event) / num_iterations
    

    print(f"Setup: Batch={batch_size}, SeqLen={seq_len}, Dim={dim}")
    #print(f"Average Sliding Window Atten time: {avg_sw_time_ms:.3f} ms")
    print(f"Average Flash Atten time: {avg_flash_time_ms:.3f} ms")
    print(f"Average Naive Atten time: {avg_naive_time_ms:.3f} ms")

    #print(f"Sliding Window Speedup: {avg_flash_time_ms / avg_sw_time_ms:.3f}x")
    print(f"Flash Speedup: {avg_naive_time_ms / avg_flash_time_ms:.3f}x")

if __name__ == '__main__':
    benchmark()