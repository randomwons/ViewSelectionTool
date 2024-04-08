#ifndef __INTRINSIC_H__
#define __INTRINSIC_H__

#include <string>
#include <fstream>
#include <sstream>

class Intrinsic {
public:
    double data[9];
    double* d_data;

    Intrinsic(const std::string& filepath) {
        std::ifstream file(filepath);
        std::string line;
        std::string cell;
        int index = 0;
        if(!file.is_open()) {
            std::string errorMsg = "[Intrinsic] Can't open file" + filepath;
            std::runtime_error(errorMsg.c_str());
            return;
        }

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while(iss >> cell) {
                if(index < 9) {
                    data[index++] = std::stod(cell);
                }
            }
        }
        file.close();  
    }
    ~Intrinsic() {
        if(d_data) cudaFree(d_data);
    }

    double* cudaData() {
        if(!d_data) {
            cudaMalloc((void**)&d_data, sizeof(double) * 9);
            cudaMemcpy(d_data, data, sizeof(double) * 9, cudaMemcpyHostToDevice);
        }
        return d_data;
    }

    double fx() const { return data[0]; }
    double fy() const { return data[4]; }
    double cx() const { return data[2]; }
    double cy() const { return data[5]; }

    void print() const {
        for(int i = 0; i < 3; i++){
            printf("%f, %f, %f\n", data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]);
        }
    }

};

#endif // __INTRINSIC_H__