#include <cuda_runtime.h>
#include <vector>


#include "buffer.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"
#include "kernel.cuh"
#include "octree.h"
#include "node.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


inline int splitNode(std::vector<Node>& nodes, int nodeIdx) {
    if(!nodes[nodeIdx].isLeaf) return -1;
    int idx = (int)nodes.size();
    for(int i = 0; i < 8; i++){
        nodes.push_back(Node(nodeIdx, -1, true));
    }
    nodes[nodeIdx].isLeaf = false;
    nodes[nodeIdx].firstChildIdx = idx;
    return idx;
}

int main() {

    int width = 534;
    int height = 720;
    int N = 160;
    dim3 gridLayout(width / 32 + 1, height / 32 + 1, N);
    dim3 blockLayout(32, 32);

    Buffer<Intrinsic> intrinsic(Intrinsic("C:/DATASET/armadillo/intrinsic/0.txt"));
    thrust::host_vector<Pose> h_poses(N);
    for(int i = 0; i < N; i++) {
        std::string filepath = "C:/DATASET/armadillo/pose/" + std::to_string(i) + ".txt";
        h_poses[i] = Pose(filepath);
    }
    Buffer<Pose> poses(h_poses);
    Buffer<Ray> rays(width * height * N);

    Buffer<Octree> d_octree(1);
    setOctree<<<1, 1>>>(d_octree.get(), make_double3(-0.64, -0.64, -0.64), make_double3(0.64, 0.64, 0.64), 0.01);

    std::vector<Node> nodes;
    nodes.push_back(Node(-1, -1, true));
    int foo1 = splitNode(nodes, 0);
    for(int a = 0; a < 8; a++){
        int foo2 = splitNode(nodes, foo1 + a);
        for(int b = 0; b < 8; a++){
            int foo3 = splitNode(nodes, foo2 + b);
        }
    }
    Buffer<Node> bufNode(nodes);

    Buffer<uchar4> frames(width * height * N);
    generateRays<<<gridLayout, blockLayout>>>(rays.get(), poses.get(), intrinsic.get(), width, height, N);
    raytracing<<<gridLayout, blockLayout>>>(rays.get(), frames.get(), d_octree.get(), bufNode.get(), width, height, N);
    cudaDeviceSynchronize();

    for(int i = 0; i < N; i++){
        std::string filename = "data/test" + std::to_string(i) + ".png";
        stbi_write_png(filename.c_str(), width, height, 4, (void*)(frames.toCPU() + i * width * height), width * 4);

    }

}