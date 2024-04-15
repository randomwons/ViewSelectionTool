#ifndef __DEPTH_H__
#define __DEPTH_H__

#include <cuda_runtime.h>
#include "npy.hpp"
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

class DepthNpy {
public:
    __host__ DepthNpy() = default;
    __host__ DepthNpy(const std::string& filepath) {
        npy::npy_data<double> d = npy::read_npy<double>(filepath);
        // data = d.data;
        data = d.data.data();

        std::vector<unsigned long> shape = d.shape;
        width_ = shape[1];
        height_ = shape[0];
    }

    __host__ __device__ int width() const { return width_; }
    __host__ __device__ int height() const { return height_; }

private:
    double* data;
    // std::vector<double> data;
    int width_, height_;

};


#endif // __DEPTH_H__