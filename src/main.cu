#include <cuda_runtime.h>

#include "buffer.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"
#include "kernel.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {

    int width = 534;
    int height = 720;
    int N = 160;

    Buffer<Intrinsic> intrinsic(Intrinsic("C:/DATASET/armadillo/intrinsic/0.txt"));
    Buffer<Ray> rays(width * height * N);
    thrust::host_vector<Pose> h_poses(N);
    for(int i = 0; i < N; i++) {
        std::string filepath = "C:/DATASET/armadillo/pose/" + std::to_string(i) + ".txt";
        h_poses[i] = Pose(filepath);
    }
    Buffer<Pose> poses(h_poses);

    dim3 gridLayout(width / 32 + 1, height / 32 + 1, N);
    dim3 blockLayout(32, 32);

    Buffer<uchar4> frames(width * height * N);

    generateRays<<<gridLayout, blockLayout>>>(rays.get(), poses.get(), intrinsic.get(), width, height, N);
    raytracing<<<gridLayout, blockLayout>>>(rays.get(), frames.get(), width, height, N);
    cudaDeviceSynchronize();

    thrust::host_vector<uchar4> test(width * height * N);
    thrust::copy(frames.d_data, frames.d_data + N * width * height, test.begin());

    for(int i = 0; i < N; i++){
        std::string filename = "data/test" + std::to_string(i) + ".png";
        stbi_write_png(filename.c_str(), width, height, 4, (void*)(frames.toCPU() + i * width * height), width * 4);

    }

}