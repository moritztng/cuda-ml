#pragma once

#include <string>

class Tensor;

void read_image(const std::string& path, Tensor& tensor, int& height, int& width);
void write_image(const std::string& path, Tensor& tensor, int height, int width);
void create_coordinates(int height, int width, Tensor& tensor);
void normalize(const std::vector<float>& max, Tensor& tensor);
