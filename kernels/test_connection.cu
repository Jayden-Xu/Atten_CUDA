
// test the connection between pytorch and cuda

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_one_kernel(float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        a[idx] += 1.0f;
    }
}

torch::Tensor add_one_cuda(torch::Tensor input) {
    auto output = input.clone();
    int n = output.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_one_kernel<<<blocks, threads>>>(output.data_ptr<float>(), n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def( // m: module object
        "add_one", // function accessible from python
        &add_one_cuda, // pointer to the actual c++ function
        "add one" // docstring
    );
}