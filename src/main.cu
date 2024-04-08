#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "loader/depth.h"
#include "loader/image.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"

struct Ray {
    double3 o, d;
    __device__ __host__ Ray(double3 o, double3 d) : o(o), d(d) {}
};

class Octree {
public:
    float3 min, max, center;
    float resolution;

    __device__ __host__ Octree(float3 min, float3 max, float resolution) 
        : min(min), max(max), resolution(resolution) {
            center = make_float3((max.x - min.x) / 2, (max.y - min.y) / 2, (max.z - min.z) / 2); 
         }

    __device__ bool traverse(const Ray& ray) {
        
        double3 center = make_double3(0, 0, 0);
        double radius = 0.1;

        double3 oc = make_double3(ray.o.x - center.x, ray.o.y - center.y, ray.o.z - center.z);
        double a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        double b = 2. * (oc.x * ray.d.x + oc.y * ray.d.y + oc.z * ray.d.z);
        double c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - radius * radius;

        double dis = b*b - 4*a*c;
        return (dis >= 0);
       
    }

};

__global__ void generateRays(Ray* rays, int width, int height, double* intrinsic, double* pose){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= width || y >= height) return;

    double fx = intrinsic[0];
    double fy = intrinsic[4];
    double cx = intrinsic[2];
    double cy = intrinsic[5];

    // double fx = 1000.;
    // double fy = 1000.;
    // double cx = 640.;
    // double cy = 360.;

    double dx = (x - cx) / fx;
    double dy = (y - cy) / fy;
    double dz = 1;

    // double norm = norm3d(dx, dy, dz);
    // dx /= norm;
    // dy /= norm;
    // dz /= norm;

    dx = pose[0 * 4 + 0] * dx + pose[1 * 4 + 0] * dy + pose[2 * 4 + 0] * dz;
    dy = pose[0 * 4 + 1] * dx + pose[1 * 4 + 1] * dy + pose[2 * 4 + 1] * dz;
    dz = pose[0 * 4 + 2] * dx + pose[1 * 4 + 2] * dy + pose[2 * 4 + 2] * dz;
    // dx = pose[0 * 4 + 0] * dx + pose[0 * 4 + 1] * dy + pose[0 * 4 + 2] * dz;
    // dy = pose[1 * 4 + 0] * dx + pose[1 * 4 + 1] * dy + pose[1 * 4 + 2] * dz;
    // dz = pose[2 * 4 + 0] * dx + pose[2 * 4 + 1] * dy + pose[2 * 4 + 2] * dz;
        double norm = norm3d(dx, dy, dz);
    dx /= norm;
    dy /= norm;
    dz /= norm;
    // double ox = pose[0 * 4 + 3];
    // double oy = pose[1 * 4 + 3];
    // double oz = pose[2 * 4 + 3];
    double ox = -(pose[0 * 4 + 0] * pose[0 * 4 + 3] + pose[1 * 4 + 0] * pose[1 * 4 + 3] + pose[2 * 4 + 0] * pose[2 * 4 + 3]);
    double oy = -(pose[0 * 4 + 1] * pose[0 * 4 + 3] + pose[1 * 4 + 1] * pose[1 * 4 + 3] + pose[2 * 4 + 1] * pose[2 * 4 + 3]);
    double oz = -(pose[0 * 4 + 2] * pose[0 * 4 + 3] + pose[1 * 4 + 2] * pose[1 * 4 + 3] + pose[2 * 4 + 2] * pose[2 * 4 + 3]);

    int pid = y * width + x;
    rays[pid].o = make_double3(ox, oy, oz);
    rays[pid].d = make_double3(dx, dy, dz);

    if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        for(int i = 0; i < 4; i++){
            printf("[CUDA] %f, %f, %f, %f\n", pose[i * 4 + 0], pose[i * 4 + 1], pose[i * 4 + 2], pose[i * 4 + 3]);
        }
        printf("[INV] %f, %f, %f, %f\n", pose[0 * 4 + 0], pose[1 * 4 + 0], pose[2 * 4 + 0], ox);
        printf("[INV] %f, %f, %f, %f\n", pose[0 * 4 + 1], pose[1 * 4 + 1], pose[2 * 4 + 1], oy);
        printf("[INV] %f, %f, %f, %f\n", pose[0 * 4 + 2], pose[1 * 4 + 2], pose[2 * 4 + 2], oz);
        printf("[INV] %f, %f, %f, %f\n", pose[3 * 4 + 0], pose[3 * 4 + 0], pose[3 * 4 + 0], 1.);
    }

}

__global__ void raytracing(uchar4* data, Octree* octree, Ray* rays, int width, int height) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int pid = y * width + x;
    if(x >= width / 2 - 10 && x <= width /2 + 10 &&
       y >= height / 2 - 10 && y <= height /2 + 10) {
        data[pid].x = 255;
        data[pid].y = 0;
        data[pid].z = 255;
        data[pid].w = 255;
        return;    
    }


    if(octree->traverse(rays[pid])) {
        data[pid].x = 255;
        data[pid].y = 255;
        data[pid].z = 255;
        data[pid].w = 255;
        return;    
    }
    // printf("max : %f, %f, %f\n", octree->max.x, octree->max.y, octree->max.z);
    // data[pid].x = (unsigned char)value;
    data[pid].x = (unsigned char)(__saturatef(rays[pid].d.x) * 255.0f);
    data[pid].y = (unsigned char)(__saturatef(rays[pid].d.y) * 255.0f);
    data[pid].z = (unsigned char)(__saturatef(rays[pid].d.z) * 255.0f);
    data[pid].w = 255;
}

// __global__ void setOctree(Octree* octree) {
//     if(threadIdx.x != 0) return;

//     octree = new Octree(make_float3(0, 0, 0), make_float3(1.28, 1.28, 1.28), 0.01);
// }


int main() {

    Octree* octree = new Octree(make_float3(0, 0, 0), make_float3(1.28, 1.28, 1.28), 0.01);
    Octree* d_octree;
    cudaMalloc((void**)&d_octree, sizeof(Octree));
    cudaMemcpy(d_octree, octree, sizeof(Octree), cudaMemcpyHostToDevice);
    // setOctree<<<1, 1>>>(d_octree);

    // int N = 1;
    for(int N = 0; N < 1; N++){
    DepthNpy depth("C:/DATASET/dataset/armadillo/depth/" + std::to_string(N) + ".npy");
    Image image("C:/DATASET/dataset/armadillo/color/" + std::to_string(N) + ".png");
    Intrinsic intrinsic("C:/DATASET/dataset/armadillo/intrinsic/" + std::to_string(N) + ".txt");
    Pose pose("C:/DATASET/dataset/armadillo/pose/" + std::to_string(N) + ".txt");
    pose.print();
    int width = depth.width();
    int height = depth.height();
    // int width = 1280;
    // int height = 720;

    dim3 threadLayout(32, 32);
    dim3 gridLayout(width / 32 + 1, height / 32 + 1);

    Ray* rays;
    cudaMalloc((void**)&rays, sizeof(Ray) * width * height);

    uchar4* data;
    cudaMalloc((void**)&data, sizeof(uchar4) * width * height);

    double* d_intrinsic;
    cudaMalloc((void**)&d_intrinsic, sizeof(double) * 9);
    cudaMemcpy(d_intrinsic, intrinsic.data, sizeof(double) * 9, cudaMemcpyHostToDevice);

    double* d_pose;
    // double t_pose[] = {
    //     1, 0,  0, 0.1,
    //     0, 1,  0, 0.0,
    //     0, 0,  1, 1,
    //     0, 0, 0, 1
    // };
    double t_pose[] = {
        0.0000000, -1.0000000,  0.0000000, 0.1,
        1.0000000,  0.0000000,  0.0000000, 0.0,
        0.0000000,  0.0000000,  1.0000000, 1,
        0, 0, 0, 1
    };
    cudaMalloc((void**)&d_pose, sizeof(double) * 16);
    // cudaMemcpy(d_pose, pose.data, sizeof(double) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pose, t_pose, sizeof(double) * 16, cudaMemcpyHostToDevice);

    generateRays<<<gridLayout, threadLayout>>>(rays, width, height, d_intrinsic, d_pose);
    raytracing<<<gridLayout, threadLayout>>>(data, d_octree, rays, width, height);
    cudaDeviceSynchronize();

    uchar4* output = new uchar4[width * height];
    cudaMemcpy(output, data, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

    stbi_write_png(("outputtest" + std::to_string(N) + ".png").c_str(), width, height, 4, (void*)output, width * 4);

    delete[] output;
    cudaFree(data);
    cudaFree(rays);
    cudaFree(d_pose);
    cudaFree(d_intrinsic);
    }


    cudaFree(d_octree);
    return 0;

}
