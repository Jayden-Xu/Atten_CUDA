
import torch
from torch.utils.cpp_extension import load
import os

from pytorch_reference import dense_attention_pytorch

naive_atten = load(
    name = "naive_atten",
    sources = [os.path.join(os.path.dirname(__file__), '../kernels/naive_atten.cu')],
    extra_cflags = ['-O3'],
    verbose = True
)

def test_naive_atten():

    batch_size, seq_len, dim = 2, 128, 64

    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda')
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda')
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda')
    
    true_output = dense_attention_pytorch(Q, K, V)
    
    result = naive_atten.naive_atten(Q, K, V)

    error = (result - true_output).abs().max().item()

    if error > 1e-5:
        print(f"Error: {error}")
    else:
        print("Success")


def benchmark_naive_atten():

    batch_size, seq_len, dim = 2, 128, 64
    num_warmup, num_iterations = 10, 100

    Q = torch.randn(batch_size, seq_len, dim, device = 'cuda')
    K = torch.randn(batch_size, seq_len, dim, device = 'cuda')
    V = torch.randn(batch_size, seq_len, dim, device = 'cuda')

    # test naive atten kernel speed
    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    for _ in range(num_warmup):
        _ = naive_atten.naive_atten(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iterations):
        _ = naive_atten.naive_atten(Q, K, V)

    end_event.record()
    torch.cuda.synchronize()

    total_naive_time_ms = start_event.elapsed_time(end_event)
    avg_naive_time_ms = total_naive_time_ms / num_iterations

    # benchmark against pytorch
    for _ in range(num_warmup):
        _ = dense_attention_pytorch(Q, K, V)
    
    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iterations):
        _ = dense_attention_pytorch(Q, K, V)
    
    end_event.record()
    torch.cuda.synchronize()

    total_pytorch_time_ms = start_event.elapsed_time(end_event)
    avg_pytorch_time_ms = total_pytorch_time_ms / num_iterations

    print(f"Average naive atten time: {avg_naive_time_ms:.2f} ms")
    print(f"Average pytorch atten time: {avg_pytorch_time_ms:.2f} ms")

if __name__ == '__main__':
    test_naive_atten()
    benchmark_naive_atten()

