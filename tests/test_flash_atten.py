
import torch
from torch.utils.cpp_extension import load
import os

from pytorch_reference import dense_attention_pytorch

cuda_source = os.path.join(os.path.dirname(__file__), '../kernels/flash_atten.cu')

flash_atten = load(
    name = "flash_atten",
    sources = [cuda_source],
    extra_cflags = ['-O3'],
    verbose = True
)

def test_flash_atten():
    print("Testing FlashAttention V2 correctness...")
    batch_size, seq_len, dim = 4, 1024, 64 

    torch.manual_seed(0)
    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)

    true_output = dense_attention_pytorch(Q, K, V)
    
    result = flash_atten.flash_atten(Q, K, V)

    error = (result - true_output).abs().max().item()

    if error > 1e-4:
        print(f"Error: {error}")
    else:
        print("Success")


def benchmark_flash_atten():

    batch_size, seq_len, dim = 16, 4096, 64
    num_warmup, num_iterations = 10, 100

    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda', dtype=torch.float32)

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    for _ in range(num_warmup):
        _ = flash_atten.flash_atten(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = flash_atten.flash_atten(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()

    avg_flash_time_ms = start_event.elapsed_time(end_event) / num_iterations
    
    for _ in range(num_warmup):
        _ = dense_attention_pytorch(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iterations):
        _ = dense_attention_pytorch(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()

    avg_pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations

    print(f"Setup: Batch={batch_size}, SeqLen={seq_len}, Dim={dim}")
    print(f"Average Flash time: {avg_flash_time_ms:.3f} ms")
    print(f"Average PyTorch (dense) time: {avg_pytorch_time_ms:.3f} ms")

if __name__ == '__main__':
    test_flash_atten()
    benchmark_flash_atten()
