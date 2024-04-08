#ifndef __POSE_H__
#define __POSE_H__

#include <string>
#include <fstream>
#include <sstream>

class Pose {
public:
    double data[16];
    double* d_data;

    Pose(const std::string& filepath) {
        std::ifstream file(filepath);
        std::string line;
        std::string cell;
        int index = 0;
        if(!file.is_open()) {
            std::string errorMsg = "[Pose] Can't open file" + filepath;
            std::runtime_error(errorMsg.c_str());
            return;
        }

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while(iss >> cell) {
                if(index < 16) {
                    data[index++] = std::stod(cell);
                }
            }
        }
        file.close();  
    }
    ~Pose() {
        if(d_data) cudaFree(d_data);
    }

    double* cudaData() {
        if(!d_data) {
            cudaMalloc((void**)&d_data, sizeof(double) * 16);
            cudaMemcpy(d_data, data, sizeof(double) * 16, cudaMemcpyHostToDevice);
        }
        return d_data;
    }

    void print() const {
        for(int i = 0; i < 4; i++){
            printf("%f, %f, %f, %f\n", data[i * 4 + 0], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]);
        }
    }
};

#endif // __POSE_H__