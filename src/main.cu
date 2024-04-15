#include <cuda_runtime.h>
#include <vector>


#include "buffer.h"

#include "loader/intrinsic.h"
#include "loader/pose.h"
#include "loader/depth.h"

#include "kernel.cuh"
#include "octree.h"
#include "node.h"

constexpr int SELECT_VIEW = 20;

int main() {

    int width = 534;
    int height = 720;
    int N = 160;
    bool save = true;

    dim3 gridLayout(width / 32 + 1, height / 32 + 1, N);
    dim3 blockLayout(32, 32);

    // Load Intrinsic
    Buffer<Intrinsic> intrinsic(Intrinsic("C:/DATASET/dataset/armadillo/intrinsic/0.txt"));

    // Load Poses
    thrust::host_vector<Pose> h_poses(N);
    for(int i = 0; i < N; i++) {
        std::string filepath = "C:/DATASET/dataset/armadillo/pose/" + std::to_string(i) + ".txt";
        h_poses[i] = Pose(filepath);
    }
    Buffer<Pose> poses(h_poses);

    // // Load Depth
    thrust::host_vector<DepthNpy> h_depthes(N);
    for(int i = 0; i < N; i++) {
        std::string filepath = "C:/DATASET/dataset/armadillo/depth/" + std::to_string(i) + ".npy";
        h_depthes[i] = DepthNpy(filepath);
    }
    Buffer<DepthNpy> depthes(h_depthes);
    
    // Rays and output frames allocate
    Buffer<Ray> rays(width * height * N);

    // Octree allocate and initialize
    Buffer<Octree> d_octree(1);
    setOctree<<<1, 1>>>(d_octree.get(), make_double3(-0.64, -0.64, -0.64), make_double3(0.64, 0.64, 0.64), 0.01);
    
    // Node allocat and initialize
    printf("[Node] create node!\n");
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Node> nodes;
    buildNode(nodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("[Node] Done!\n");
    printf("[Node] size : %d\n", (int)nodes.size());
    printf("[Node] Elapsed time : %.1lf ms\n", elapsed.count());
    Buffer<Node> bufNode(nodes);
    //////////

    //// Kernel // Ray casting
    printf("[Kernel] start!\n");
    cudaEvent_t kernelStart, kernelEnd;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventRecord(kernelStart);

    // Generate rays and casting to octree 
    generateRays<<<gridLayout, blockLayout>>>(rays.get(), poses.get(), intrinsic.get(), width, height, N);

    // update octree
    // updateOctree<<<gridLayout, blockLayout>>>(rays.get(), depthes.get(), d_octree.get(), width, height, N);


    raycastToOctree<<<gridLayout, blockLayout>>>(rays.get(), d_octree.get(), bufNode.get(), width, height, N);

    // Get Max index;
    Buffer<float> outputs(N);
    sumRayValues<<<gridLayout, blockLayout>>>(rays.get(), outputs.get(), width, height, N);
    auto maxIter = thrust::max_element(outputs.get(), outputs.get() + N);
    int maxIdx = maxIter - outputs.get();

    Buffer<uchar4> frames;
    if(save) {
        frames.alloc(width * height * N);
        std::vector<double> maxValues(N);
        thrust::device_ptr<double> d_values(reinterpret_cast<double*>(rays.get()) + offsetof(Ray, value));
        for(int i = 0; i < N; i++){
            double maxValue = thrust::reduce(d_values + width * height * i, d_values + width * height * (i + 1), -std::numeric_limits<double>::infinity(), thrust::maximum<double>());
            maxValues[i] = maxValue;
        }
        Buffer<double> d_maxValues(maxValues);

        colorMapFrames<<<gridLayout, blockLayout>>>(rays.get(), frames.get(), d_maxValues.get(), width, height, N);
    }

    cudaEventRecord(kernelEnd);
    cudaEventSynchronize(kernelEnd);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, kernelStart, kernelEnd);
    printf("[Kernel] Done!\n");
    printf("[Kernel] Elapsed time : %.1f ms\n", milliseconds);
    
    if(save) {
        printf("[Save] Start data writing!\n");
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < N; i++){
            std::string filename = "data/test" + std::to_string(i) + ".png";
            stbi_write_png(filename.c_str(), width, height, 4, (void*)(frames.toCPU() + i * width * height), width * 4);
            // if (i == 3) break;
        }
        end = std::chrono::high_resolution_clock::now();
        printf("[Save] Done!\n");
        elapsed = end - start;
        printf("[Save] Elapsed time : %.1lf ms\n", elapsed.count());
    }
    return 0;
}