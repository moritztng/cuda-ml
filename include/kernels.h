#pragma once

__global__ void fill_scalar(size_t n, float scalar, float* output);
__global__ void add(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* sum);
__global__ void subtract(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* difference);
__global__ void subtract(size_t n, float* tensor1, float* tensor2, float* difference);
__global__ void multiply(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* product);
__global__ void divide(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* quotient);
__global__ void matrix_multiply(size_t rank, size_t height, size_t width, size_t shared_dim, size_t* tensor1_strides, size_t* tensor2_strides, float* tensor1, float* tensor2, float* matrix_product);
__global__ void negate(size_t n, float* input, float* output);
__global__ void square(size_t n, float* input, float* output);
__global__ void sum(size_t n, float* input, float* output);
__global__ void batch_sum(size_t n, size_t batch_size, size_t stride, float* input, float* output);
__global__ void relu(size_t n, float* input, float* output);
__global__ void relu_d(size_t n, float* input, float* output);
