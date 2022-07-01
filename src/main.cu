#include "tensor.h"
#include "data.h"
#include "network.h"
#include "loss.h"
#include "optimizer.h"


int main()
{
    Tensor targets{};
    Tensor coordinates{};
    int height{};
    int width{};
    read_image("image.png", targets, height, width);
    create_coordinates(height, width, coordinates);
    targets = targets / Tensor::from_scalar(255, {1, 1});
    coordinates = coordinates / Tensor::from_vector({static_cast<float>(height - 1), static_cast<float>(width - 1)}, {1, 2});
    MultiLayerPerceptron network{2, {64, 1}};
    StochasticGradientDescent optimizer{network.parameters(), 0.01};
    const size_t data_size{ static_cast<size_t>(coordinates.shape[0]) };
    const size_t batch_size{ 32 };
    const size_t n_epochs{ 1000 };
    const size_t n_batches{ data_size / batch_size + 1 };
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        for (int batch = 0; batch < n_batches; ++batch) {
            Tensor coordinates_batch{};
            Tensor targets_batch{};
            get_batch(batch, batch_size, coordinates, targets, coordinates_batch, targets_batch);
            const Tensor predictions{ network(coordinates_batch) };
            const Tensor loss{ mean_squared_error(predictions, targets_batch) };
            loss.backward();
            optimizer.step();
            optimizer.zero_gradients();
            std::cout << "loss " << loss[{0, 0}] << '\n';
        }
    }
    network.detach();
    Tensor predictions{ network(coordinates) };
    write_image("reconstruction.png", predictions, height, width);
    return 0;
}
