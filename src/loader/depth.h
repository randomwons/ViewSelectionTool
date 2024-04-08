#ifndef __DEPTH_H__
#define __DEPTH_H__

#include "npy.hpp"
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

class DepthNpy {
public:
    DepthNpy(const std::string& filepath) {
        npy::npy_data<double> d = npy::read_npy<double>(filepath);
        data = d.data;

        std::vector<unsigned long> shape = d.shape;
        width_ = shape[1];
        height_ = shape[0];
    }
    void save_as_png(const std::string& savepath) {
        
        double max = -1;
        for(int i = 0; i < data.size(); i++){
            if(max < data[i]) max = data[i];
        }
        std::vector<uint8_t> image(data.size());
        for(int i = 0; i < image.size(); i++){
            image[i] = (uint8_t)(data[i] / max * 255.f);
        }

        stbi_write_png(savepath.c_str(), width_, height_, 1, image.data(), width_);
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    std::vector<double> data;
    int width_, height_;

};


#endif // __DEPTH_H__