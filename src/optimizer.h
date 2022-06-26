#pragma once

#include <vector>
#include "tensor.h"

class Optimizer {
public:
    const std::vector<Tensor*> parameters{};
    const Tensor learning_rate{};
    Optimizer(const std::vector<Tensor*>& parameters, float learning_rate);
    virtual void step() = 0;
    void zero_gradients() const;
};

class StochasticGradientDescent : public Optimizer {
public:
    StochasticGradientDescent(const std::vector<Tensor*>& parameters, float learning_rate);
    virtual void step();
};
