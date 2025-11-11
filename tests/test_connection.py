
import torch
from torch.utils.cpp_extension import load
import os

print("Compiling CUDA extension")
test_cuda = load(
    name = "test_connection",
    sources = [os.path.join(os.path.dirname(__file__), '../kernels/test_connection.cu')],
    extra_cuda_cflags = ['-O3'],
    verbose = True
)

print("Testing connection")
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device = 'cuda')
print(f"Input: {x}")

y = test_cuda.add_one(x)
print(f"Output: {y}")

expected = x + 1
error = (y - expected).abs().max().item()

if error > 1e-5:
    print(f"Error: {error}")
else:
    print("Success")

