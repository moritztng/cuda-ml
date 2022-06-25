#include <cmath>
#include "network.h"

std::vector<Tensor*> Module::parameters() {
    return {};
}

void Module::requires_gradients() {
    for (Tensor* parameter : parameters()) {
        parameter->requires_gradients();
    }
}

Linear::Linear(size_t input_dim, size_t output_dim) :
    weights{ Tensor::random_normal(0, std::sqrt(2. / input_dim), {static_cast<int>(input_dim), static_cast<int>(output_dim)}) },
    bias{ Tensor::from_scalar(0, {1, static_cast<int>(output_dim)}) } {}

Tensor Linear::operator() (const Tensor& input) const {
    return mm(input, weights) + bias;
}

std::vector<Tensor*> Linear::parameters() {
    return {&weights, &bias};
}

Tensor ReLU::operator() (const Tensor& input) const {
    return relu(input);
}

MultiLayerPerceptron::MultiLayerPerceptron(size_t input_layer_dim, std::initializer_list<size_t> layer_dims) {
    size_t input_dim{ input_layer_dim };
    for (size_t output_dim : layer_dims) {
        const Linear linear_layer{ input_dim, output_dim };
        linear_layers.push_back(linear_layer);
        input_dim = output_dim;
    }
}

Tensor MultiLayerPerceptron::operator() (const Tensor& input) const {
    Tensor output{ input };
    for (Linear linear_layer : linear_layers) {
        output = linear_layer(output);
        output = relu_layer(output);
    }
    return output;
}

std::vector<Tensor*> MultiLayerPerceptron::parameters() {
    std::vector<Tensor*> parameters{};
    for (int i = 0; i < linear_layers.size(); ++i) {
        const std::vector<Tensor*> linear_layer_parameters{ linear_layers[i].parameters() };
        parameters.insert(parameters.end(), linear_layer_parameters.begin(), linear_layer_parameters.end());
    }
    return parameters;
}
