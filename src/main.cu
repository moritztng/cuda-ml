#include "tensor.h"
#include <iostream>

int main()
{
    Tensor<float> tensor{ Tensor<float>::from_scalar(0, {3, 3, 3}) };
    std::cout << tensor;
    return 0;
}
