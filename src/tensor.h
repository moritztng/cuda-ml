#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <random>
#include <iostream>

std::random_device device;
std::mt19937 random_number_generator{ device() };

template <typename T>
__global__
void add(size_t n, T* tensor1, T* tensor2, T* sum)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sum[i] = tensor1[i] + tensor2[i];
}

template <typename T>
__global__
void subtract(size_t n, T* tensor1, T* tensor2, T* difference)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) difference[i] = tensor1[i] - tensor2[i];
}

template <typename T>
__global__
void multiply(size_t n, T* tensor1, T* tensor2, T* product)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) product[i] = tensor1[i] * tensor2[i];
}

template <typename T>
__global__
void divide(size_t n, T* tensor1, T* tensor2, T* quotient)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) quotient[i] = tensor1[i] / tensor2[i];
}

template <typename T>
class Tensor {
private:
    T* data{};
    Tensor(const std::vector<int>& shape) :
        shape{ shape },
        rank{ shape.size() },
        strides( rank ),
        n_elements{ std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) },
        size{ n_elements * sizeof(T) }
    {
        cudaMalloc(&data, size);
        size_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
public:
    std::vector<int> shape{};
    const size_t rank{};
    std::vector<size_t> strides{};
    const size_t n_elements{};
    const size_t size{};
    T operator[] (const std::vector<int>& indices) {
        T scalar;
        const size_t index{ std::inner_product(strides.begin(), strides.end(), indices.begin(), static_cast<size_t>(0)) };
        cudaMemcpy(&scalar, data + index, sizeof(T), cudaMemcpyDeviceToHost);
        return scalar;
    }
    friend Tensor<T> operator+ (const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        Tensor<T> sum{ tensor1.shape };
        add<T><<<(sum.n_elements + 255) / 256, 256>>>(sum.n_elements, tensor1.data, tensor2.data, sum.data);
        return sum;
    }
    friend Tensor<T> operator- (const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        Tensor<T> difference{ tensor1.shape };
        subtract<T><<<(difference.n_elements + 255) / 256, 256>>>(difference.n_elements, tensor1.data, tensor2.data, difference.data);
        return difference;
    }
    friend Tensor<T> operator* (const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        Tensor<T> product{ tensor1.shape };
        multiply<T><<<(product.n_elements + 255) / 256, 256>>>(product.n_elements, tensor1.data, tensor2.data, product.data);
        return product;
    }
    friend Tensor<T> operator/ (const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        Tensor<T> quotient{ tensor1.shape };
        divide<T><<<(quotient.n_elements + 255) / 256, 256>>>(quotient.n_elements, tensor1.data, tensor2.data, quotient.data);
        return quotient;
    }
    friend std::ostream& operator<< (std::ostream& out, Tensor<T>& tensor) {
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
        for (size_t dim: tensor.shape) out << dim << ", ";
        out << "), ";
        out << "rank = " << tensor.rank << ", ";
        out << "strides = (";
        for (size_t stride: tensor.strides) out << stride << ", ";
        out << "), ";
        out << "n_elements = " << tensor.n_elements << ", ";
        out << "size = " << tensor.size << ", ";
        out << '\n';
        return out;
    }
    static Tensor<T> from_vector(const std::vector<T>& vector, const std::vector<int>& shape) {
        Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::copy(vector.begin(), vector.end(), array);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;
    }
    static Tensor<T> from_scalar(T scalar, const std::vector<int>& shape) {
        Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::fill_n(array, tensor.n_elements, scalar);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
    static Tensor<T> random_uniform(T min, T max, const std::vector<int>& shape) {
        Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::uniform_real_distribution<T> distribution{ min, max };
        for (int i = 0; i < tensor.n_elements; ++i) {
            array[i] = distribution(random_number_generator);
        }
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
    static Tensor<T> random_normal(T mean, T standard_deviation, const std::vector<int>& shape) {
        Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::normal_distribution<T> distribution{ mean, standard_deviation };
        for (int i = 0; i < tensor.n_elements; ++i) {
            array[i] = distribution(random_number_generator);
        }
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;
    }
};
