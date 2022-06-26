#include "tensor.h"
#include "network.h"
#include "loss.h"
#include "optimizer.h"

int main()
{
    const Tensor inputs{Tensor::random_normal(0, 1, {1, 3})};
    const Tensor targets{Tensor::random_normal(0, 1, {1, 3})};
    MultiLayerPerceptron network{3, {6, 12, 6, 3}};
    network.requires_gradients();
    StochasticGradientDescent optimizer{network.parameters(), 0.01};
    for (int i = 0; i < 3; ++i) {
        const Tensor predictions{ network(inputs) };
        const Tensor loss{ mean_squared_error(predictions, targets) };
        loss.backward();
        optimizer.step();
        optimizer.zero_gradients();
    }
    return 0;
}
