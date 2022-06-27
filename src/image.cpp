#include <png.h>
#include <string>
#include "image.h"
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
        for (int x = 0; x < width * 3; ++x) {
            vector.push_back(static_cast<float>(row[x]));    
        }
    }
    tensor = Tensor::from_vector(vector, {height * width, 3});
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
