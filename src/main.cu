#include "tensor.h"

int main()
{
    Tensor<float>::from_scalar(0, {10, 10});
    return 0;
}
