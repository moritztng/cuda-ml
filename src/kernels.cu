#include "kernels.h"

__global__
void fill_scalar(size_t n, float scalar, float* output)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) output[index] = scalar;
}

__device__
void get_indices(size_t index, size_t rank, size_t* strides1, size_t* strides2, size_t* strides, size_t* indices)
{
    size_t index_remainder = index;
    indices[0] = 0;
    indices[1] = 0;
    for (int i = 0; i < rank; ++i) {
        const size_t dim = index_remainder / strides[i];
        index_remainder -= dim * strides[i];
        indices[0] += dim * strides1[i];
        indices[1] += dim * strides2[i];
    }
}

__global__
void add(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* sum)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
      size_t indices[2];
      get_indices(index, rank, tensor1_strides, tensor2_strides, strides, indices);
      sum[index] = tensor1[indices[0]] + tensor2[indices[1]];
  }
}

__global__
void subtract(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* difference)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
      size_t indices[2];
      get_indices(index, rank, tensor1_strides, tensor2_strides, strides, indices);
      difference[index] = tensor1[indices[0]] - tensor2[indices[1]];
  }
}

__global__
void subtract(size_t n, float* tensor1, float* tensor2, float* difference)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) difference[index] = tensor1[index] - tensor2[index];
}

__global__
void multiply(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* product)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
      size_t indices[2];
      get_indices(index, rank, tensor1_strides, tensor2_strides, strides, indices);
      product[index] = tensor1[indices[0]] * tensor2[indices[1]];
  }
}

__global__
void divide(size_t n, size_t rank, size_t* tensor1_strides, size_t* tensor2_strides, size_t* strides, float* tensor1, float* tensor2, float* quotient)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
      size_t indices[2];
      get_indices(index, rank, tensor1_strides, tensor2_strides, strides, indices);
      quotient[index] = tensor1[indices[0]] / tensor2[indices[1]];
  }
}

__global__
void matrix_multiply(size_t rank, size_t height, size_t width, size_t shared_dim, size_t* tensor1_strides, size_t* tensor2_strides, float* tensor1, float* tensor2, float* matrix_product)
{
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t column = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && column < width) {
        const size_t tensor1_start = blockIdx.z * height * shared_dim + row * tensor1_strides[rank - 2];
        const size_t tensor2_start = blockIdx.z * width * shared_dim + column * tensor2_strides[rank - 1];
        float product{ 0 };
        for (int i = 0; i < shared_dim; ++i) {
            product += tensor1[tensor1_start + i * tensor1_strides[rank - 1]] * tensor2[tensor2_start + i * tensor2_strides[rank - 2]];
        }
        matrix_product[blockIdx.z * height * width + row * width + column] = product;
    }
}

__global__
void negate(size_t n, float* input, float* output)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = -input[i];
}

__global__
void square(size_t n, float* input, float* output)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) output[index] = input[index] * input[index];
}

__global__
void sum(size_t n, float* input, float* output)
{
  float sum{ 0 };
  for (int i = 0; i < n; ++i) {
      sum += input[i];
  }
  output[0] = sum;
}

__global__
void relu(size_t n, float* input, float* output)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i] > 0 ? input[i] : 0;
}

__global__
void relu_d(size_t n, float* input, float* output)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i] > 0 ? 1 : 0;
}
