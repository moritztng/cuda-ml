#include "tensor.h"
#include "network.h"
#include "loss.h"
#include "optimizer.h"

int main()
{
    const Tensor inputs{Tensor::from_scalar(1, {10, 3})};
    const Tensor targets{Tensor::from_scalar(2, {10, 1})};
    MultiLayerPerceptron network{3, {6, 1}};
    StochasticGradientDescent optimizer{network.parameters(), 0.01};
    for (int i = 0; i < 3; ++i) {
        const Tensor predictions{ network(inputs) };
        const Tensor loss{ mean_squared_error(predictions, targets) };
        loss.backward();
        std::cout << "Loss: " << loss << "Predictions: " << predictions;
        for (Tensor* parameter : network.parameters()) {
            std::cout << (*parameter).gradients();
        }
        optimizer.step();
        optimizer.zero_gradients();
    }
    return 0;
}
