#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Image {
public:
    Image(const std::string& filepath) {

        data = stbi_load(filepath.c_str(), &width_, & height_, &channel_, 0);
        if(!data) {
            printf("[Image] Failed to open file %s\n", filepath);
        }

    }
    ~Image() {
        if(data) stbi_image_free(data);
    }

    int width() { return width_; }
    int height() { return height_; }
    int channel() { return channel_; }

private:
    unsigned char* data;
    int width_, height_, channel_;
};

#endif // __IMAGE_H__