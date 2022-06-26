#include <vector>
#include "loss.h"
#include "tensor.h"

Tensor mean_squared_error(const Tensor& prediction, const Tensor& target) {
    const Tensor scalar{ Tensor::from_scalar(1. / prediction.n_elements, std::vector<int>(prediction.rank, 1)) };
    return scalar * sum(square(prediction - target));
}
