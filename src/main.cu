#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_free.h>

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
    double3 min, max, center;
    float resolution;

    __device__ __host__ Octree(double3 min, double3 max, float resolution) 
        : min(min), max(max), resolution(resolution) {
            center = make_double3((max.x + min.x) / 2, (max.y + min.y) / 2, (max.z + min.z) / 2); 
         }

    __device__ bool traverse(Ray& ray) {

        if(ray.d.x < 0.0f){
            ray.o.x = center.x * 2.0f - ray.o.x;
            ray.d.x = -ray.d.x;
        }
        if(ray.d.y < 0.0f){
            ray.o.y = center.y * 2.0f - ray.o.y;
            ray.d.y = -ray.d.y;
        }
        if(ray.d.z < 0.0f){
            ray.o.z = center.z * 2.0f - ray.o.z;
            ray.d.z = -ray.d.z;
        }
    
        const double tx0 = (ray.d.x == 0) ? DBL_MIN : (min.x - ray.o.x) * (1 / ray.d.x);
        const double tx1 = (ray.d.x == 0) ? DBL_MAX : (max.x - ray.o.x) * (1 / ray.d.x);
        const double ty0 = (ray.d.y == 0) ? DBL_MIN : (min.y - ray.o.y) * (1 / ray.d.y);
        const double ty1 = (ray.d.y == 0) ? DBL_MAX : (max.y - ray.o.y) * (1 / ray.d.y);
        const double tz0 = (ray.d.z == 0) ? DBL_MIN : (min.z - ray.o.z) * (1 / ray.d.z);
        const double tz1 = (ray.d.z == 0) ? DBL_MAX : (max.z - ray.o.z) * (1 / ray.d.z);
        
        if(fmax(fmax(tx0, ty0), tz0) < fmin(fmin(tx1, ty1), tz1)) return true;
        return false;
       
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

    double dx = (x - cx) / fx;
    double dy = (y - cy) / fy;
    double dz = 1;

    double worlddx = pose[0 * 4 + 0] * dx + pose[1 * 4 + 0] * dy + pose[2 * 4 + 0] * dz;
    double worlddy = pose[0 * 4 + 1] * dx + pose[1 * 4 + 1] * dy + pose[2 * 4 + 1] * dz;
    double worlddz = pose[0 * 4 + 2] * dx + pose[1 * 4 + 2] * dy + pose[2 * 4 + 2] * dz;

    double norm = norm3d(worlddx, worlddy, worlddz);
    worlddx /= norm;
    worlddy /= norm;
    worlddz /= norm;

    double ox = -(pose[0 * 4 + 0] * pose[0 * 4 + 3] + pose[1 * 4 + 0] * pose[1 * 4 + 3] + pose[2 * 4 + 0] * pose[2 * 4 + 3]);
    double oy = -(pose[0 * 4 + 1] * pose[0 * 4 + 3] + pose[1 * 4 + 1] * pose[1 * 4 + 3] + pose[2 * 4 + 1] * pose[2 * 4 + 3]);
    double oz = -(pose[0 * 4 + 2] * pose[0 * 4 + 3] + pose[1 * 4 + 2] * pose[1 * 4 + 3] + pose[2 * 4 + 2] * pose[2 * 4 + 3]);

    int pid = y * width + x;
    rays[pid].o = make_double3(ox, oy, oz);
    rays[pid].d = make_double3(worlddx, worlddy, worlddz);

}

__global__ void raytracing(uchar4* data, thrust::device_ptr<Octree*> octree, Ray* rays, int width, int height) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int pid = y * width + x;
    if(((Octree*)(*octree))->traverse(rays[pid])) {
        data[pid].x = 255;
        data[pid].y = 255;
        data[pid].z = 255;
        data[pid].w = 255;
        return;    
    }

    data[pid].x = (unsigned char)(__saturatef(rays[pid].d.x) * 255.0f);
    data[pid].y = (unsigned char)(__saturatef(rays[pid].d.y) * 255.0f);
    data[pid].z = (unsigned char)(__saturatef(rays[pid].d.z) * 255.0f);
    data[pid].w = 255;
}

__global__ void setOctree(thrust::device_ptr<Octree*> octree) {
    if(threadIdx.x != 0) return;

    *octree = new Octree(make_double3(-0.5, -0.5, -0.5), make_double3(0.5, 0.5, 0.5), 0.01);

}

int main() {

    thrust::device_ptr<Octree*> d_octree;
    d_octree = thrust::device_new<Octree*>();
    setOctree<<<1, 1>>>(d_octree);

    for(int N = 0; N < 160; N++){
        DepthNpy depth("C:/DATASET/armadillo/depth/" + std::to_string(N) + ".npy");
        Image image("C:/DATASET/armadillo/color/" + std::to_string(N) + ".png");
        Intrinsic intrinsic("C:/DATASET/armadillo/intrinsic/" + std::to_string(N) + ".txt");
        Pose pose("C:/DATASET/armadillo/pose/" + std::to_string(N) + ".txt");

        int width = depth.width();
        int height = depth.height();

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
        cudaMalloc((void**)&d_pose, sizeof(double) * 16);
        cudaMemcpy(d_pose, pose.data, sizeof(double) * 16, cudaMemcpyHostToDevice);

        generateRays<<<gridLayout, threadLayout>>>(rays, width, height, d_intrinsic, d_pose);
        raytracing<<<gridLayout, threadLayout>>>(data, d_octree, rays, width, height);
        cudaDeviceSynchronize();

        uchar4* output = new uchar4[width * height];
        cudaMemcpy(output, data, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

        stbi_write_png(("data/outputtest" + std::to_string(N) + ".png").c_str(), width, height, 4, (void*)output, width * 4);

        delete[] output;
        cudaFree(data);
        cudaFree(rays);
        cudaFree(d_pose);
        cudaFree(d_intrinsic);
    }
    thrust::device_free(d_octree);
    return 0;

}
