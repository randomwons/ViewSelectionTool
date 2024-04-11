#ifndef __POSE_H__
#define __POSE_H__

#include <cuda_runtime.h>

class Pose {

    static constexpr size_t NUM_ELEMENTS = 16;

public:
    double data[NUM_ELEMENTS];

    __host__ Pose() = default;
    __host__ Pose(const std::string& filename) {

        std::ifstream file(filename);
        if(!file.is_open()) {
            std::string errorMsg = "[Pose] Can't open file" + filename;
            std::runtime_error(errorMsg.c_str());
            return;
        }

        std::string line;
        std::string cell;
        int index = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while(iss >> cell) {
                if(index < NUM_ELEMENTS) {
                    data[index++] = std::stod(cell);
                }
            }
        }
        file.close();

    }

    __device__ void print() const {
        for(int i = 0; i < NUM_ELEMENTS; i++){
            printf("%lf\n", data[i]);
        }
    }

};

#endif // __POSE_H__