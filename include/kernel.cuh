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

    rays[pid] = Ray(make_double3(ox, oy, oz), make_double3(tdx, tdy, tdz));

}

__global__ void raytracing(Ray* rays, uchar4* frames, Octree* octree, Node* nodes, int width, int height, int N) {

    int x = XCOORD;
    int y = YCOORD;
    int z = ZCOORD;
    if(x >= width || y >= height || z >= N) return;

    int pid = z * width * height + y * width + x;

    if(octree->traverse(rays[pid], nodes)){

        frames[pid].x = 255;
        frames[pid].y = 255;
        frames[pid].z = 255;
        frames[pid].w = 255;
        
        return;
    }


    frames[pid].x = (unsigned char)(__saturatef(rays[pid].d.x + 1) * 255.0f);
    frames[pid].y = (unsigned char)(__saturatef(rays[pid].d.y + 1) * 255.0f);
    frames[pid].z = (unsigned char)(__saturatef(rays[pid].d.z + 1) * 255.0f);
    frames[pid].w = 255;

}

__global__ void setOctree(Octree* octree, double3 min, double3 max, double resolution) {
    if(threadIdx.x != 0) return;
    
    *octree = Octree(min, max, resolution);
}


#endif // __KERNEL_CUH__