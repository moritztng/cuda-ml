#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <random>
#include <iostream>

std::random_device device;
std::mt19937 random_number_generator{ device() };

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
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        cudaMalloc(&data, size);
    }
public:
    std::vector<int> shape{};
    const size_t rank{};
    size_t offset{};
    std::vector<int> strides{};
    const size_t n_elements{};
    const size_t size{};
    T scalar() {
        T scalar;
        cudaMemcpy(&scalar, data + offset, sizeof(T), cudaMemcpyDeviceToHost);
        return scalar;
    }
    Tensor<T> operator[] (const std::vector<int>& indices) {
        const size_t index{ std::inner_product(strides.begin(), strides.end(), indices.begin(), static_cast<size_t>(0)) };
        Tensor<T> tensor{std::vector<int>(1, rank)};
        tensor.data = data;
        tensor.offset = index;
        return tensor;
    }
    friend std::ostream& operator<< (std::ostream& out, Tensor<T>& tensor) {
        std::vector<int> indices(tensor.rank, 0);
        out << std::string(tensor.rank, '[');
        for (int i = 0; i < tensor.n_elements; ++i) {
            out << tensor[indices].scalar() << ", ";
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
        return out;
    };
    static Tensor<T> from_vector(const std::vector<T>& vector, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::copy(vector.begin(), vector.end(), array);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;
    }
    static Tensor<T> from_scalar(T scalar, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        T* array = (T*)malloc(tensor.size);
        std::fill_n(array, tensor.n_elements, scalar);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
    static Tensor<T> random_uniform(T min, T max, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
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
        const Tensor<T> tensor{ shape };
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
