#include <vector>
#include <numeric>
#include <functional>

template <typename T>
class Tensor {
private:
    T* data {};
    std::vector<int> shape {};
public:
    Tensor(T* array, std::vector<int> shape) : shape{ shape } {
        int n_elements{ std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) };
        size_t size = n_elements * sizeof(T);
        cudaMalloc(&data, size);
        cudaMemcpy(data, array, size, cudaMemcpyHostToDevice);
    }
};
