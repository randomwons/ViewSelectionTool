#ifndef __RAY_H__
#define __RAY_H__

#include <cuda_runtime.h>

struct Ray {

    double3 o, d;

    __host__ __device__ Ray(double3 o, double3 d) : o(o), d(d) { }

};


#endif // __RAY_H__