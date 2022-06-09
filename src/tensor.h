#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <random>

template <typename T>
class Tensor {
private:
    static std::mt19937 random_number_generator;
    T* data{};
    Tensor(const std::vector<int>& shape) : 
        shape{ shape }, 
        n_elements{ std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) },
        size{ n_elements * sizeof(T) } {
            cudaMalloc(&data, size);
    }
public:
    const std::vector<int> shape{};
    const int n_elements{};
    const size_t size{};
    static Tensor<T> from_vector(const std::vector<T>& vector, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        const T* array = (T*)malloc(tensor.size);
        std::copy(vector.begin(), vector.end(), array);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;
    }
    static Tensor<T> from_scalar(T scalar, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        const T* array = (T*)malloc(tensor.size);
        std::fill_n(array, tensor.n_elements, scalar);
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
    static Tensor<T> random_uniform(T min, T max, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        const T* array = (T*)malloc(tensor.size);
        const std::uniform_real_distribution<T> distribution{ min, max };
        for (int i = 0; i < n_elements; ++i) {
            array[i] = distribution(random_number_generator);
        }
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
    static Tensor<T> random_normal(T mean, T standard_deviation, const std::vector<int>& shape) {
        const Tensor<T> tensor{ shape };
        const T* array = (T*)malloc(tensor.size);
        const std::normal_distribution<T> distribution{ mean, standard_deviation };
        for (int i = 0; i < n_elements; ++i) {
            array[i] = distribution(random_number_generator);
        }
        cudaMemcpy(tensor.data, array, tensor.size, cudaMemcpyHostToDevice);
        free(array);
        return tensor;    
    }
};

template <typename T>
std::mt19937 Tensor<T>::random_number_generator{ std::random_device() };
