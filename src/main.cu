#include "tensor.h"
#include "image.h"
#include "network.h"
#include "loss.h"
#include "optimizer.h"

int main()
{
    Tensor rgbs{};
    Tensor coordinates{};
    int height{};
    int width{};
    read_image("image.png", rgbs, height, width);
    create_coordinates(height, width, coordinates);
    MultiLayerPerceptron network{2, {6, 3}};
    StochasticGradientDescent optimizer{network.parameters(), 0.01};
    for (int i = 0; i < 3; ++i) {
        const Tensor predictions{ network(coordinates) };
        const Tensor loss{ mean_squared_error(predictions, rgbs) };
        loss.backward();
        std::cout << "Loss: " << loss << "Predictions: " << predictions;
        optimizer.step();
        optimizer.zero_gradients();
    }
    network.detach();
    Tensor predictions{ network(coordinates) };
    write_image("reconstruction.png", predictions, height, width);
    return 0;
}
