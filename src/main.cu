#include "tensor.h"
#include <iostream>

int main()
{
    Tensor<float> tensor{ Tensor<float>::random_normal(1, 2, {5, 5}) };
    std::cout << tensor[{1, 1}].offset;
    return 0;
}
