#include <png.h>
#include <string>
#include <vector>
#include "data.h"
#include "tensor.h"

void read_image(const std::string& path, Tensor& tensor, int& height, int& width) {
    FILE* file_pointer = fopen(path.c_str(), "rb");
    png_structp png_pointer = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_pointer = png_create_info_struct(png_pointer);
    png_init_io(png_pointer, file_pointer);
    png_read_info(png_pointer, info_pointer);
    height = png_get_image_height(png_pointer, info_pointer);
    width = png_get_image_width(png_pointer, info_pointer);
    png_bytep* row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; ++y)
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_pointer, info_pointer));
    png_read_image(png_pointer, row_pointers);
    std::vector<float> vector{};
    for (int y = 0; y < height; ++y) {
        png_byte* row = row_pointers[y];
        for (int x = 0; x < width; ++x) {
            vector.push_back(static_cast<float>(row[x]));
        }
    }
    tensor = Tensor::from_vector(vector, {height * width, 1});
}

void write_image(const std::string& path, Tensor& tensor, int height, int width) {
    FILE* file_pointer = fopen(path.c_str(), "wb");
    png_structp png_pointer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_pointer = png_create_info_struct(png_pointer);
    png_init_io(png_pointer, file_pointer);
    png_set_IHDR(png_pointer, info_pointer, width, height,
        8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_pointer, info_pointer);
    png_bytep* row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; ++y) {
        png_byte* row = (png_byte*) malloc(png_get_rowbytes(png_pointer, info_pointer));
        row_pointers[y] = row;
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<png_byte>(tensor[{y * width + x, 0}]);
        }
    }
    png_write_image(png_pointer, row_pointers);
    png_write_end(png_pointer, NULL);
    for (int y = 0; y < height; ++y)
        free(row_pointers[y]);
    free(row_pointers);
    fclose(file_pointer);
}

void create_coordinates(int height, int width, Tensor& tensor) {
    std::vector<float> vector{};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            vector.insert(vector.end(), {static_cast<float>(y), static_cast<float>(x)});
        }
    }
    tensor = Tensor::from_vector(vector, {height * width, 2});
}

void get_batch(size_t index, size_t batch_size, const Tensor& inputs, const Tensor& outputs, Tensor& inputs_batch, Tensor& outputs_batch) {
    std::vector<int> inputs_shape{ inputs.shape };
    const int rest = inputs_shape[0] - index * batch_size;
    const int batch_n = batch_size < rest ? batch_size : rest;
    inputs_shape[0] = batch_n;
    inputs_batch = Tensor(inputs_shape);
    cudaMemcpy(inputs_batch.data.get(), inputs.data.get() + index * batch_size * inputs.strides[0], inputs_batch.size, cudaMemcpyDeviceToDevice);
    std::vector<int> outputs_shape{ outputs.shape };
    outputs_shape[0] = batch_n;
    outputs_batch = Tensor(outputs_shape);
    cudaMemcpy(outputs_batch.data.get(), outputs.data.get() + index * batch_size * outputs.strides[0], outputs_batch.size, cudaMemcpyDeviceToDevice);
}