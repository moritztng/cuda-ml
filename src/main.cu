#include "tensor.h"
#include <vector>
#include <numeric>
#include <functional>

int main()
{
    std::vector<int> shape{ 10, 10 };
    int n_elements{ std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) };
    size_t size = n_elements * sizeof(float);
    float* array = (float*)malloc(size);
    Tensor<float> tensor{ array, shape };
    return 0;
}
