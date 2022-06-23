#include "tensor.h"

int main()
{
    Tensor tensor{ Tensor::from_scalar(1, {2, 2}) };
    std::cout << tensor;
    return 0;
}
