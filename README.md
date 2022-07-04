## Installation
```bash
cmake . -B build
cmake --build build
sudo make install -C build
```

## Learning an Image
```cpp
#include <cuda-ml/cuda-ml.h>

Tensor targets{};
Tensor coordinates{};
int height{};
int width{};
read_image("image.png", targets, height, width);
create_coordinates(height, width, coordinates);
normalize({255}, targets);
normalize({static_cast<float>(height - 1), static_cast<float>(width - 1)}, coordinates);

MultiLayerPerceptron network{2, {64, 1}};
StochasticGradientDescent optimizer{network.parameters(), 0.01};

const size_t n_epochs{ 1000 };
const size_t print_epochs{ 100 };
for (int epoch = 0; epoch < n_epochs; ++epoch) {
    const Tensor predictions{ network(coordinates) };
    const Tensor loss{ mean_squared_error(predictions, targets) };
    loss.backward();
    optimizer.step();
    optimizer.zero_gradients();
    if ((epoch + 1) % print_epochs == 0)
        std::cout << "epoch " << epoch << " loss " << loss[{0, 0}] << '\n';
}

network.detach();
Tensor predictions{ network(coordinates) };
normalize({1 / 255}, predictions);
write_image("reconstruction.png", predictions, height, width);
```

## Tensor Class
### Construction
```cpp
Tensor::from_scalar(1, {2, 2})
Tensor::from_vector({1, 2, 3, 4}, {2, 2})
Tensor::random_uniform(0, 1, {2, 2})
Tensor::random_normal(0, 1, {2, 2})
```

### Operations
```cpp
tensor1 + tensor2
tensor1 - tensor2
tensor1 * tensor2
tensor1 / tensor2
mm(tensor1, tensor2)
square(tensor)
relu(tensor)
sum(tensor)
```

### Indexing
```cpp
tensor[{1, 2}]
```

### Automatic Differentiation
```cpp
tensor1.requires_gradient();
product = tensor1 * tensor2;
product.backward();
gradients = tensor1.gradients();
tensor1.detach();
```

### Output
```cpp
std::cout << Tensor::from_vector({1, 2, 3, 4}, {2, 2});
```
