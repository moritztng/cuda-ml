#pragma once

#include <vector>
#include <initializer_list>
#include "tensor.h"

class Module 
{
public:
    virtual Tensor operator() (const Tensor& input) const = 0;
    virtual std::vector<Tensor*> parameters();
};

class Linear : public Module
{
public:
    Tensor weights{};
    Tensor bias{};
    Linear(size_t input_dim, size_t output_dim, bool requires_gradients = true);
    virtual Tensor operator() (const Tensor& input) const;
    virtual std::vector<Tensor*> parameters();
};

class ReLU : public Module
{
public:
    virtual Tensor operator() (const Tensor& input) const;
};

class MultiLayerPerceptron : public Module
{
public:
    std::vector<Linear> linear_layers{};
    const ReLU relu_layer{};
    MultiLayerPerceptron(size_t input_layer_dim, std::initializer_list<size_t> layer_dims, bool requires_gradients = true);
    virtual Tensor operator() (const Tensor& input) const;
    virtual std::vector<Tensor*> parameters();
};
