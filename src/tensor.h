#pragma once

#include <vector>
#include <memory>
#include <iostream>

class Backward;

class Tensor {
public:
    std::vector<int> shape{};
    size_t rank{};
    std::vector<size_t> strides{};
    size_t n_elements{};
    size_t size{};
    std::shared_ptr<float> data{};
    std::shared_ptr<Backward> backward_pointer{};

    Tensor();
    Tensor(const std::vector<int>& shape);
    static Tensor from_scalar(float scalar, const std::vector<int>& shape);
    static Tensor from_vector(const std::vector<float>& vector, const std::vector<int>& shape);
    static Tensor random_uniform(float min, float max, const std::vector<int>& shape);
    static Tensor random_normal(float mean, float standard_deviation, const std::vector<int>& shape);

    float operator[] (const std::vector<int>& indices) const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    void requires_gradients(bool sum = false);
    Tensor detach() const;
    void backward() const;
    void backward(const Tensor& gradients) const;
    Tensor& gradients() const;

    void fill(float scalar);
    Tensor& operator-= (const Tensor& tensor);
    friend Tensor operator- (const Tensor& input);
    friend Tensor operator+ (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator- (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator* (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator/ (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor mm (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor relu (const Tensor& input);
    friend Tensor relu_d (const Tensor& input);
    friend Tensor square(const Tensor& input);
    friend Tensor sum (const Tensor& input);
    friend Tensor batch_sum (const Tensor& input);

    friend std::ostream& operator<< (std::ostream& out, const Tensor& tensor);
};
