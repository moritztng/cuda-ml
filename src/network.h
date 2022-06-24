#include <vector>
#include <initializer_list>
#include "tensor.h"

class Module 
{
public:
    virtual Tensor operator() (const Tensor& input) const = 0;
};

class Linear : Module
{
public:
    Tensor weights{};
    Tensor bias{};
    Linear(size_t input_dim, size_t output_dim);
    virtual Tensor operator() (const Tensor& input) const;
};

class ReLU : Module
{
public:
    virtual Tensor operator() (const Tensor& input) const;
};

class MultiLayerPerceptron : Module
{
public:
    std::vector<Linear> linear_layers{};
    const ReLU relu_layer{};
    MultiLayerPerceptron(size_t input_layer_dim, std::initializer_list<size_t> layer_dims);
    virtual Tensor operator() (const Tensor& input) const;
};
