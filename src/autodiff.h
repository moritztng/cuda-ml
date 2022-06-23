#pragma once

#include <vector>

class Tensor;

class Backward
{
public:
    std::vector<Tensor> tensors{};
    const std::vector<Backward*> backwards{};
    Backward();
    Backward(const std::vector<Backward*>& backwards);
    Backward(const std::vector<Tensor>& tensors, const std::vector<Backward*>& backwards);
    virtual void operator() (const Tensor& gradients);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class AccumulateGradients : public Backward
{
public:
    virtual void operator() (const Tensor& gradients);
};

class AddBackward : public Backward
{
public:
    AddBackward(const std::vector<Backward*>& backwards);
};

class SubtractBackward : public Backward
{
public:
    SubtractBackward(const std::vector<Backward*>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class MultiplyBackward : public Backward
{
public:
    MultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<Backward*>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class MatrixMultiplyBackward : public Backward
{
public:
    MatrixMultiplyBackward(const std::vector<Tensor>& tensors, const std::vector<Backward*>& backwards);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class NegateBackward : public Backward
{
public:
    NegateBackward(Backward* backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};

class SumBackward : public Backward
{
public:
    SumBackward(Backward* backward);
};

class ReluBackward : public Backward
{
public:
    ReluBackward(const Tensor& tensor, Backward* backward);
private:
    virtual Tensor backward(const Tensor& gradients, size_t input_index) const;
};
