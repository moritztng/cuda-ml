#include <vector>
#include "optimizer.h"
#include "tensor.h"

Optimizer::Optimizer(const std::vector<Tensor*>& parameters, float learning_rate) : 
    parameters{ parameters },
    learning_rate{ Tensor::from_scalar(learning_rate, std::vector<int>(parameters[0]->rank, 1)) } {}

void Optimizer::zero_gradients() const {
    for (Tensor* parameter : parameters) {
        (*parameter).gradients().fill(0);
    }
}

StochasticGradientDescent::StochasticGradientDescent(const std::vector<Tensor*>& parameters, float learning_rate) : Optimizer{ parameters, learning_rate } {}

void StochasticGradientDescent::step() {
    for (Tensor* parameter : parameters) {
        *parameter -= learning_rate * (*parameter).gradients();
    }
}
