#include <vector>
#include "utils.h"
#include "tensor.h"

void prepare_broadcast(const Tensor& tensor1, const Tensor& tensor2, size_t** d_tensor1_strides, size_t** d_tensor2_strides, size_t** d_strides, Tensor& sum) {
    std::vector<int> shape{ tensor1.shape };
    size_t tensor1_strides[tensor1.rank]{ 0 };
    size_t tensor2_strides[tensor2.rank]{ 0 };
    for (int i = 0; i < tensor1.rank; ++i) {
        if (tensor1.shape[i] == tensor2.shape[i]) {
            tensor1_strides[i] = tensor1.strides[i];
            tensor2_strides[i] = tensor2.strides[i];
        }
        else if (tensor1.shape[i] > tensor2.shape[i]) {
            tensor1_strides[i] = tensor1.strides[i];
        }
        else {
            tensor2_strides[i] = tensor2.strides[i];
            shape[i] = tensor2.shape[i];
        }

    }
    sum = Tensor{ shape };
    const size_t strides_size = sum.rank * sizeof(size_t);
    cudaMalloc(d_strides, strides_size);
    cudaMalloc(d_tensor1_strides, strides_size);
    cudaMalloc(d_tensor2_strides, strides_size);
    cudaMemcpy(*d_strides, &sum.strides[0], strides_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_tensor1_strides, tensor1_strides, strides_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_tensor2_strides, tensor2_strides, strides_size, cudaMemcpyHostToDevice);
}

float* dataMalloc(size_t size) {
    float* data;
    cudaMalloc(&data, size);
    return data;
}
