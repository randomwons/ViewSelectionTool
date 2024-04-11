#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_free.h>

template <typename T>
class Buffer {
public:
    Buffer(size_t size) : size(size) {
        d_data = thrust::device_malloc<T>(size);
    }
    Buffer(T& data) : size(1) {
        d_data = thrust::device_malloc<T>(1);
        thrust::copy(&data, &data + 1, d_data);
    }
    Buffer(thrust::host_vector<T>& data) {
        d_data = thrust::device_malloc<T>(data.size());
        thrust::copy(data.begin(), data.end(), d_data);
    }
    Buffer(std::vector<T>& data) {
        d_data = thrust::device_malloc<T>(data.size());
        thrust::copy(data.begin(), data.end(), d_data);
    }

    ~Buffer() {
        if(d_data) thrust::device_free(d_data);
    }

    T* get() const { return d_data.get(); }

    T* toCPU() {
        cpudata = thrust::host_vector<T>(size);
        thrust::copy(d_data, d_data + size, cpudata.begin());
        return cpudata.data();
    }


// private:
    thrust::host_vector<T> cpudata;
    thrust::device_ptr<T> d_data;
    size_t size;

};



#endif // __BUFFER_H__