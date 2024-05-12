#ifndef __DEPTH_H__
#define __DEPTH_H__

#include "npy/npy.hpp"
#include <string>

class Depth {
public:
    Depth(const std::string& filepath) {
        npy::npy_data<double> d = npy::read_npy<double>(filepath);
        data = d.data;
        std::vector<unsigned long> shape = d.shape;
        width_ = shape[1];
        height_ = shape[0];
    }

    inline int width() const { return width_; }
    inline int height() const { return height_; }

// private:
    std::vector<double> data;

    int width_, height_;
};

#endif // __DEPTH_H__