#pragma once

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <random>
#include <iostream>

class Backward;

class Tensor {
private:
    float* data{};
public:
    std::vector<int> shape{};
    size_t rank{};
    std::vector<size_t> strides{};
    size_t n_elements{};
    size_t size{};
    Backward* backward{};
    Tensor();
    Tensor(const std::vector<int>& shape);
    float operator[] (const std::vector<int>& indices);
    Tensor operator-() const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    Tensor gradients() const;
    void requires_gradients();
    friend Tensor operator+ (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator- (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator* (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor operator/ (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor mm (const Tensor& tensor1, const Tensor& tensor2);
    friend Tensor relu (const Tensor& input);
    friend Tensor relu_d (const Tensor& input);
    friend Tensor sum (const Tensor& input);
    friend std::ostream& operator<< (std::ostream& out, Tensor& tensor);
    static Tensor from_vector(const std::vector<float>& vector, const std::vector<int>& shape);
    static Tensor from_scalar(float scalar, const std::vector<int>& shape);
    static Tensor random_uniform(float min, float max, const std::vector<int>& shape);
    static Tensor random_normal(float mean, float standard_deviation, const std::vector<int>& shape);
};

void prepare_broadcast(const Tensor& tensor1, const Tensor& tensor2, size_t** d_tensor1_strides, size_t** d_tensor2_strides, size_t** d_strides, Tensor& sum);
