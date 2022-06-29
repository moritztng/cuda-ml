#pragma once

#include <vector>
#include <memory>

class Tensor;

class Backward
{
public:
    std::vector<Tensor> tensors{};
    const std::vector<std::shared_ptr<Backward>> backwards{};
    Backward();
    Backward(const std::vector<std::shared_ptr<Backward>>& backwards);
    Backward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards);
    virtual void operator() (const Tensor& gradients);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class AccumulateGradients : public Backward
{
public:
    const bool sum{};
    AccumulateGradients(bool sum = false);
    virtual void operator() (const Tensor& gradients);
};

class AddBackward : public Backward
{
public:
    AddBackward(const std::vector<std::shared_ptr<Backward>>& backwards);
};

class SubtractBackward : public Backward
{
public:
    SubtractBackward(const std::vector<std::shared_ptr<Backward>>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class MultiplyBackward : public Backward
{
public:
    MultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class MatrixMultiplyBackward : public Backward
{
public:
    MatrixMultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<std::shared_ptr<Backward>>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class NegateBackward : public Backward
{
public:
    NegateBackward(std::shared_ptr<Backward> backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class SquareBackward : public Backward
{
public:
    SquareBackward(const Tensor& tensor, std::shared_ptr<Backward> backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class SumBackward : public Backward
{
public:
    const std::vector<int> shape{};
    SumBackward(const std::vector<int>& shape, std::shared_ptr<Backward> backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;    
};

class ReluBackward : public Backward
{
public:
    ReluBackward(const Tensor& tensor, std::shared_ptr<Backward> backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};
