#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "ray.h"
#include "octree.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"
#include "node.h"

#define XCOORD blockDim.x * blockIdx.x + threadIdx.x;
#define YCOORD blockDim.y * blockIdx.y + threadIdx.y;
#define ZCOORD blockIdx.z;

__global__ void generateRays(Ray* rays, Pose* poses, Intrinsic* intrinsic, int width, int height, int N) {

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD;
    if(x >= width || y >= height || z >= N) return;

    double dx = (x - intrinsic->cx()) / intrinsic->fx();
    double dy = (y - intrinsic->cy()) / intrinsic->fy();
    double dz = 1;

    double tdx = poses[z].data[0] * dx + poses[z].data[4] * dy + poses[z].data[8] * dz;
    double tdy = poses[z].data[1] * dx + poses[z].data[5] * dy + poses[z].data[9] * dz;
    double tdz = poses[z].data[2] * dx + poses[z].data[6] * dy + poses[z].data[10] * dz;

    double ox = poses[z].data[0] * poses[z].data[3] + poses[z].data[4] * poses[z].data[7] + poses[z].data[8] * poses[z].data[11];
    double oy = poses[z].data[1] * poses[z].data[3] + poses[z].data[5] * poses[z].data[7] + poses[z].data[9] * poses[z].data[11];
    double oz = poses[z].data[2] * poses[z].data[3] + poses[z].data[6] * poses[z].data[7] + poses[z].data[10] * poses[z].data[11];
    

    int pid = z * width * height + y * width + x;

    rays[pid] = Ray(make_double3(-ox, -oy, -oz), make_double3(tdx, tdy, tdz));

}

__global__ void sumRayValues(const Ray* rays, float* output, int width, int height, int N) {

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD;
    if(x >= width || y >= height || z >= N) return;

    int pid = z * width * height + y * width + z;
    atomicAdd(&output[z], (float)rays[pid].value);
    // atomicAdd(output[z], rays[pid].value);

}

__global__ void raycastToOctree(Ray* rays, Octree* octree, Node* nodes, int width, int height, int N) {

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD;
    if(x >= width || y >= height || z >= N) return;

    int pid = z * width * height + y * width + x;
    int value = octree->traverse(rays[pid], nodes);

}

__global__ void setOctree(Octree* octree, double3 min, double3 max, double resolution) {
    if(threadIdx.x != 0) return;
    
    *octree = Octree(min, max, resolution);
}


__global__ void colorMapFrames(const Ray* rays, uchar4* frames, const double* maxValues, int width, int height, int N) {

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD
    if(x >= width || y >= height || z >= N) return;

    double maxValue = maxValues[z];
    
    int pid = z * width * height + y * width + x;
    double value = rays[pid].value;
    if(value == 0.0){
        frames[pid] = make_uchar4(0, 0, 0, 255);
        return;
    }
    value /= maxValue;

    const float n = 4;
    const float v = fmaxf(0, fminf(n, value * n));
    unsigned char r = 255 * fminf(fmaxf(1.5 - fabs(v - 3), 0), 1);
    unsigned char g = 255 * fminf(fmaxf(1.5 - fabs(v - 2), 0), 1);
    unsigned char b = 255 * fminf(fmaxf(1.5 - fabs(v - 1), 0), 1);
    frames[pid] = make_uchar4(b, g, r, 255);

}

__global__ void updateOctree(Ray* rays, DepthNpy* depthes, Octree* octree, int width, int height, int N){

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD;
    if(x >= width || y >= height || z >= N) return;

    


}


#endif // __KERNEL_CUH__