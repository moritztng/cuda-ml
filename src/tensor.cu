#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <functional>
#include "tensor.h"
#include "kernels.h"
#include "autodiff.h"
#include "utils.h"

std::random_device device;
std::mt19937 random_number_generator{ device() };

Tensor::Tensor() = default;
Tensor::Tensor(const std::vector<int>& shape) :
    shape{ shape },
    rank{ shape.size() },
    strides( rank ),
    n_elements{ std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) },
    size{ n_elements * sizeof(float) },
    data{ dataMalloc(size), [](float* data){cudaFree(data);} }
{
    size_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

Tensor Tensor::from_scalar(float scalar, const std::vector<int>& shape) {
    Tensor tensor{ shape };
    float* array = (float*)malloc(tensor.size);
    std::fill_n(array, tensor.n_elements, scalar);
    cudaMemcpy(tensor.data.get(), array, tensor.size, cudaMemcpyHostToDevice);
    free(array);
    return tensor;    
}

Tensor Tensor::from_vector(const std::vector<float>& vector, const std::vector<int>& shape) {
    Tensor tensor{ shape };
    float* array = (float*)malloc(tensor.size);
    std::copy(vector.begin(), vector.end(), array);
    cudaMemcpy(tensor.data.get(), array, tensor.size, cudaMemcpyHostToDevice);
    free(array);
    return tensor;
}

Tensor Tensor::random_uniform(float min, float max, const std::vector<int>& shape) {
    Tensor tensor{ shape };
    float* array = (float*)malloc(tensor.size);
    std::uniform_real_distribution<float> distribution{ min, max };
    for (int i = 0; i < tensor.n_elements; ++i) {
        array[i] = distribution(random_number_generator);
    }
    cudaMemcpy(tensor.data.get(), array, tensor.size, cudaMemcpyHostToDevice);
    free(array);
    return tensor;    
}

Tensor Tensor::random_normal(float mean, float standard_deviation, const std::vector<int>& shape) {
    Tensor tensor{ shape };
    float* array = (float*)malloc(tensor.size);
    std::normal_distribution<float> distribution{ mean, standard_deviation };
    for (int i = 0; i < tensor.n_elements; ++i) {
        array[i] = distribution(random_number_generator);
    }
    cudaMemcpy(tensor.data.get(), array, tensor.size, cudaMemcpyHostToDevice);
    free(array);
    return tensor;
}

float Tensor::operator[] (const std::vector<int>& indices) const {
    float scalar;
    const size_t index{ std::inner_product(strides.begin(), strides.end(), indices.begin(), static_cast<size_t>(0)) };
    cudaMemcpy(&scalar, data.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
    return scalar;
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const {
    Tensor transpose{ *this };
    transpose.shape[dim1] = shape[dim2];
    transpose.shape[dim2] = shape[dim1];
    transpose.strides[dim1] = strides[dim2];
    transpose.strides[dim2] = strides[dim1];
    return transpose;
}

void Tensor::requires_gradients(bool sum) {
    backward_pointer = std::shared_ptr<Backward>{ new AccumulateGradients{ sum } };
}

Tensor Tensor::detach() const {
    Tensor tensor{ *this };
    tensor.backward_pointer = nullptr;
    return tensor;
}

void Tensor::backward() const {
    backward(Tensor::from_scalar(1, shape));
}

void Tensor::backward(const Tensor& gradients) const {
    (*backward_pointer)(gradients);
}

Tensor& Tensor::gradients() const {
    return backward_pointer->tensors[0];
}

void Tensor::fill (float scalar) {
    fill_scalar<<<(n_elements + 255) / 256, 256>>>(n_elements, scalar, data.get());
}

Tensor& Tensor::operator-= (const Tensor& tensor) {
    subtract<<<(n_elements + 255) / 256, 256>>>(n_elements, data.get(), tensor.data.get(), data.get());
    return *this;
}

Tensor operator- (const Tensor& input) {
    Tensor output{ input.shape };
    negate<<<(output.n_elements + 255) / 256, 256>>>(output.n_elements, input.data.get(), output.data.get());
    if (input.backward_pointer) output.backward_pointer = std::shared_ptr<Backward>{ new NegateBackward{ input.backward_pointer } };
    return output;
}

Tensor operator+ (const Tensor& tensor1, const Tensor& tensor2) {
    size_t* tensor1_strides{};
    size_t* tensor2_strides{};
    size_t* strides{};
    Tensor sum{};
    prepare_broadcast(tensor1, tensor2, &tensor1_strides, &tensor2_strides, &strides, sum);
    add<<<(sum.n_elements + 255) / 256, 256>>>(sum.n_elements, sum.rank, tensor1_strides, tensor2_strides, strides, tensor1.data.get(), tensor2.data.get(), sum.data.get());
    if (tensor1.backward_pointer || tensor2.backward_pointer) sum.backward_pointer = std::shared_ptr<Backward>{ new AddBackward{ {tensor1.backward_pointer, tensor2.backward_pointer} } };
    return sum;
}

Tensor operator- (const Tensor& tensor1, const Tensor& tensor2) {
    size_t* tensor1_strides{};
    size_t* tensor2_strides{};
    size_t* strides{};
    Tensor difference{};
    prepare_broadcast(tensor1, tensor2, &tensor1_strides, &tensor2_strides, &strides, difference);
    subtract<<<(difference.n_elements + 255) / 256, 256>>>(difference.n_elements, difference.rank, tensor1_strides, tensor2_strides, strides, tensor1.data.get(), tensor2.data.get(), difference.data.get());
    if (tensor1.backward_pointer || tensor2.backward_pointer) difference.backward_pointer = std::shared_ptr<Backward>{ new SubtractBackward{ {tensor1.backward_pointer, tensor2.backward_pointer} } };
    return difference;
}

Tensor operator* (const Tensor& tensor1, const Tensor& tensor2) {
    size_t* tensor1_strides{};
    size_t* tensor2_strides{};
    size_t* strides{};
    Tensor product{};
    prepare_broadcast(tensor1, tensor2, &tensor1_strides, &tensor2_strides, &strides, product);
    multiply<<<(product.n_elements + 255) / 256, 256>>>(product.n_elements, product.rank, tensor1_strides, tensor2_strides, strides, tensor1.data.get(), tensor2.data.get(), product.data.get());
    if (tensor1.backward_pointer || tensor2.backward_pointer) product.backward_pointer = std::shared_ptr<Backward>{ new MultiplyBackward{ {tensor1.detach(), tensor2.detach()}, {tensor1.backward_pointer, tensor2.backward_pointer} } };
    return product;
}

Tensor operator/ (const Tensor& tensor1, const Tensor& tensor2) {
    size_t* tensor1_strides{};
    size_t* tensor2_strides{};
    size_t* strides{};
    Tensor quotient{};
    prepare_broadcast(tensor1, tensor2, &tensor1_strides, &tensor2_strides, &strides, quotient);
    divide<<<(quotient.n_elements + 255) / 256, 256>>>(quotient.n_elements, quotient.rank, tensor1_strides, tensor2_strides, strides, tensor1.data.get(), tensor2.data.get(), quotient.data.get());
    if (tensor1.backward_pointer || tensor2.backward_pointer) {
        const Tensor tensor1_reciprocal{ Tensor::from_scalar(1, std::vector<int>(tensor1.rank, 1)) / tensor1.detach() };
        const Tensor tensor2_reciprocal{ Tensor::from_scalar(1, std::vector<int>(tensor2.rank, 1)) / tensor2.detach() };
        quotient.backward_pointer = std::shared_ptr<Backward>{ new MultiplyBackward{ {tensor1_reciprocal, tensor2_reciprocal}, {tensor1.backward_pointer, tensor2.backward_pointer} } };
    }
    return quotient;
}

Tensor mm(const Tensor& tensor1, const Tensor& tensor2) {
    std::vector<int> shape{ tensor1.shape };
    shape.back() = tensor2.shape.back();
    Tensor matrix_product{ shape };
    const size_t height = matrix_product.shape.end()[-2];
    const size_t width = matrix_product.shape.end()[-1];
    const size_t shared_dim = tensor1.shape.end()[-1];
    const size_t batch_size = matrix_product.n_elements / (height * width);
    size_t* tensor1_strides{};
    size_t* tensor2_strides{};
    const size_t strides_size = matrix_product.rank * sizeof(size_t);
    cudaMalloc(&tensor1_strides, strides_size);
    cudaMalloc(&tensor2_strides, strides_size);
    cudaMemcpy(tensor1_strides, &tensor1.strides[0], strides_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tensor2_strides, &tensor2.strides[0], strides_size, cudaMemcpyHostToDevice);
    dim3 block_dim(16, 16);
    dim3 grid_dim((height + block_dim.x - 1) / block_dim.x, (width + block_dim.y - 1) / block_dim.y, batch_size);
    matrix_multiply<<<grid_dim, block_dim>>>(matrix_product.rank, height, width, shared_dim, tensor1_strides, tensor2_strides, tensor1.data.get(), tensor2.data.get(), matrix_product.data.get());
    cudaFree(tensor1_strides);
    cudaFree(tensor2_strides);
    if (tensor1.backward_pointer || tensor2.backward_pointer) matrix_product.backward_pointer = std::shared_ptr<Backward>{ new MatrixMultiplyBackward{ {tensor1.detach(), tensor2.detach()}, {tensor1.backward_pointer, tensor2.backward_pointer} } };
    return matrix_product;
}

Tensor relu(const Tensor& input) {
    Tensor output{ input.shape };
    relu<<<(output.n_elements + 255) / 256, 256>>>(output.n_elements, input.data.get(), output.data.get());
    if (input.backward_pointer) output.backward_pointer = std::shared_ptr<Backward>{ new ReluBackward{ input.detach(), input.backward_pointer } };
    return output;
}

Tensor relu_d(const Tensor& input) {
    Tensor output{ input.shape };
    relu_d<<<(output.n_elements + 255) / 256, 256>>>(output.n_elements, input.data.get(), output.data.get());
    return output;
}

Tensor square(const Tensor& input) {
    Tensor output{ input.shape };
    square<<<(output.n_elements + 255) / 256, 256>>>(output.n_elements, input.data.get(), output.data.get());
    if (input.backward_pointer) output.backward_pointer = std::shared_ptr<Backward>{ new SquareBackward{ input.detach(), input.backward_pointer } };
    return output;
}

Tensor sum(const Tensor& input) {
    Tensor output{ std::vector<int>(input.rank, 1) };
    sum<<<1, 1>>>(input.n_elements, input.data.get(), output.data.get());
    if (input.backward_pointer) output.backward_pointer = std::shared_ptr<Backward>{ new SumBackward{ input.shape, input.backward_pointer } };
    return output;
}

Tensor batch_sum(const Tensor& input) {
    Tensor output{ {1, input.shape.back()} };
    batch_sum<<<(output.n_elements + 255) / 256, 256>>>(output.n_elements, input.shape[0], input.shape.back(), input.data.get(), output.data.get());
    return output;
}

std::ostream& operator<< (std::ostream& out, const Tensor& tensor) {
    std::vector<int> indices(tensor.rank, 0);
    out << std::string(tensor.rank, '[');
    for (int i = 0; i < tensor.n_elements; ++i) {
        out << tensor[indices] << ", ";
        for (int j = tensor.rank - 1; j >= 0; --j) {
            if (indices[j] < (tensor.shape[j] - 1)) {
                out << std::string(tensor.rank - 1 - j, '[');
                ++indices[j];
                break;
            } else {
                out << "]";
                indices[j] = 0;
            }
        }
    }
    out << '\n';
    out << "shape = (";
    for (int dim : tensor.shape) out << dim << ", ";
    out << ")\n";
    return out;
}
