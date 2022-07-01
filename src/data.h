#pragma once

#include <string>

class Tensor;

void read_image(const std::string& path, Tensor& tensor, int& height, int& width);
void write_image(const std::string& path, Tensor& tensor, int height, int width);
void create_coordinates(int height, int width, Tensor& tensor);
void get_batch(size_t index, size_t batch_size, const Tensor& inputs, const Tensor& outputs, Tensor& inputs_batch, Tensor& outputs_batch);
