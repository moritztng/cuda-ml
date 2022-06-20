#include "tensor.h"
#include <iostream>

int main()
{
    Tensor<float> tensor_from_scalar{ Tensor<float>::from_scalar(0, {3, 3, 3}) };
    std::cout << tensor_from_scalar;
    Tensor<float> tensor_from_vector{ Tensor<float>::from_vector({0, 1, 2, 3}, {2, 2}) };
    std::cout << tensor_from_vector;
    Tensor<float> tensor_random_uniform{ Tensor<float>::random_uniform(0, 10, {3, 2, 1}) };
    std::cout << tensor_random_uniform;
    Tensor<float> tensor_random_normal{ Tensor<float>::random_normal(0, 1, {10}) };
    std::cout << tensor_random_normal;
    std::cout << tensor_from_vector[{0, 0}] << '\n';
    Tensor<float> tensor1{ Tensor<float>::from_scalar(1, {2, 2}) };
    Tensor<float> tensor2{ Tensor<float>::from_scalar(2, {2, 2}) };
    Tensor<float> sum{ tensor1 + tensor2 };
    std::cout << sum;
    Tensor<float> difference{ tensor1 - tensor2 };
    std::cout << difference;
    Tensor<float> product{ tensor1 * tensor2 };
    std::cout << product;
    Tensor<float> quotient{ tensor1 / tensor2 };
    std::cout << quotient;
    Tensor<float> matrix_product{ mm(tensor1, tensor2) };
    std::cout << matrix_product;
    Tensor<float> relu_input{ Tensor<float>::random_uniform(-10, 10, {3, 2, 2}) };
    Tensor<float> relu_output{ relu(relu_input) };
    std::cout << relu_output;
    Tensor<float> broadcast_tensor1{ Tensor<float>::from_scalar(1, {1, 3, 3}) };
    Tensor<float> broadcast_tensor2{ Tensor<float>::from_scalar(2, {3, 3, 3}) };
    Tensor<float> broadcast_sum{ broadcast_tensor1 + broadcast_tensor2 };
    std::cout << broadcast_sum;
    Tensor<float> tensor1_backward{ Tensor<float>::from_scalar(1, {2, 2}) };
    tensor1_backward.requires_gradients();
    Tensor<float> tensor2_backward{ Tensor<float>::from_scalar(2, {2, 2}) };
    Tensor<float> product_backward{ tensor1_backward * tensor2_backward };
    Tensor<float> gradient{ Tensor<float>::from_scalar(1, {1, 1}) };
    (*product_backward.backward)(gradient);
    Tensor<float> gradients{ tensor1_backward.gradients() };
    std::cout << gradients;
    return 0;
}
