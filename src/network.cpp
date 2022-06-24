#include <cmath>
#include "network.h"

Linear::Linear(size_t input_dim, size_t output_dim) :
    weights{ Tensor::random_normal(0, std::sqrt(2. / input_dim), {static_cast<int>(input_dim), static_cast<int>(output_dim)}) },
    bias{ Tensor::from_scalar(0, {1, static_cast<int>(output_dim)}) } {}

Tensor Linear::operator() (const Tensor& input) const {
    return mm(input, weights) + bias;
}

Tensor ReLU::operator() (const Tensor& input) const {
    return relu(input);
}

MultiLayerPerceptron::MultiLayerPerceptron(size_t input_layer_dim, std::initializer_list<size_t> layer_dims) {
    size_t input_dim{ input_layer_dim };
    for (size_t output_dim : layer_dims) {
        linear_layers.push_back(Linear{ input_dim, output_dim });
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
