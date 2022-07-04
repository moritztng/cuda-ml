#pragma once

class Tensor;

void prepare_broadcast(const Tensor& tensor1, const Tensor& tensor2, size_t** d_tensor1_strides, size_t** d_tensor2_strides, size_t** d_strides, Tensor& sum);
float* dataMalloc(size_t size);
