#include "autodiff.h"
#include "tensor.h"

Backward::Backward() = default;
Backward::Backward(const std::vector<std::shared_ptr<Backward>>& backwards) : backwards{ backwards } {}
Backward::Backward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards) : tensors{ tensors }, backwards{ backwards } {}
void Backward::operator() (const Tensor& gradients) {
    for (int i = 0; i < backwards.size(); ++i) {
        if (backwards[i]) (*backwards[i])(backward(gradients, i));
    }
}
Tensor Backward::backward(const Tensor& gradients, size_t input_index) const { return gradients; }

AccumulateGradients::AccumulateGradients(bool sum) : sum{ sum } {}
void AccumulateGradients::operator() (const Tensor& gradients) {
    const Tensor accumulate_gradients{ sum ? batch_sum(gradients) : gradients };
    if (tensors.size()) tensors[0] = tensors[0] + accumulate_gradients;
    else tensors.push_back(accumulate_gradients);
}

AddBackward::AddBackward(const std::vector<std::shared_ptr<Backward>>& backwards) : Backward{ backwards } {}

SubtractBackward::SubtractBackward(const std::vector<std::shared_ptr<Backward>>& backwards) : Backward{ backwards } {}
Tensor SubtractBackward::backward(const Tensor& gradients, size_t input_index) const {
    return input_index ? -gradients : gradients;
}

MultiplyBackward::MultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards) : Backward{ backwards } {
    if (backwards[1]) this->tensors.push_back(tensors[0]); 
    if (backwards[0]) this->tensors.push_back(tensors[1]); 
}
Tensor MultiplyBackward::backward(const Tensor& gradients, size_t input_index) const {
    return tensors[input_index ? 0 : tensors.size() - 1] * gradients;
}

MatrixMultiplyBackward::MatrixMultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards) : Backward{ backwards } {
    if (backwards[1]) this->tensors.push_back(tensors[0]); 
    if (backwards[0]) this->tensors.push_back(tensors[1]); 
}
Tensor MatrixMultiplyBackward::backward(const Tensor& gradients, size_t input_index) const {
    return mm(tensors[input_index ? 0 : tensors.size() - 1].transpose(tensors[0].rank - 1, tensors[0].rank - 2), gradients);
}

NegateBackward::NegateBackward(std::shared_ptr<Backward> backward) : Backward{ {backward} } {}
Tensor NegateBackward::backward(const Tensor& gradients, size_t input_index) const {
    return -gradients;
}

SquareBackward::SquareBackward(const Tensor& tensor, std::shared_ptr<Backward> backward) : Backward{ {tensor}, {backward} } {}
Tensor SquareBackward::backward(const Tensor& gradients, size_t input_index) const {
    return Tensor::from_scalar(2, std::vector<int>(tensors[0].rank, 1)) * tensors[0] * gradients;
}

SumBackward::SumBackward(const std::vector<int>& shape, std::shared_ptr<Backward> backward) : shape{ shape }, Backward{ {backward} } {}
Tensor SumBackward::backward(const Tensor& gradients, size_t input_index) const {
    return Tensor::from_scalar(gradients[std::vector<int>(gradients.rank, 0)], shape);
}

ReluBackward::ReluBackward(const Tensor& tensor, std::shared_ptr<Backward> backward) : Backward{ {tensor}, {backward} } {}
Tensor ReluBackward::backward(const Tensor& gradients, size_t input_index) const {
    return relu_d(tensors[0]) * gradients;
}
