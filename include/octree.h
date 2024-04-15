#ifndef __OCTREE_H__
#define __OCTREE_H__

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "ray.h"
#include "node.h"

#define NODE_STACK_SIZE 10

class Octree {
public:

    struct Stack {

        struct SubtreeData {
            float tx0, ty0, tz0;
            float txm, tym, tzm;
            float tx1, ty1, tz1;
            int nodeIdx;
            int nextQuadrant;
        };

        int idx = 0;
        SubtreeData data[NODE_STACK_SIZE];

        __host__ __device__ inline void push(SubtreeData&& data) {
            this->data[idx] = data;
            idx++;
        }
        
        __host__ __device__ inline void pop() {
            idx--;
        }
        
        __host__ __device__ inline SubtreeData* top() {
            return data + (idx - 1);
        }

        __host__ __device__ inline bool isEmpty() {
            return idx <= 0;
        }

    };

    double3 min, max, center;
    double resolution;

    __host__ __device__ Octree(double3 min, double3 max, double resolution) 
            : min(min), max(max), resolution(resolution) {

        center = make_double3((max.x + min.z) / 2, (max.y + min.y) / 2, (max.z + min.z) / 2);

    }

    __device__ int traverse(Ray& ray, Node* nodes) {
        unsigned char a = 0;

        if(ray.d.x < 0.0f){
            ray.o.x = center.x * 2.0f - ray.o.x;
            ray.d.x = -ray.d.x;
            a |= 4;
        }
        if(ray.d.y < 0.0f){
            ray.o.y = center.y * 2.0f - ray.o.y;
            ray.d.y = -ray.d.y;
            a |= 2;
        }
        if(ray.d.z < 0.0f){
            ray.o.z = center.z * 2.0f - ray.o.z;
            ray.d.z = -ray.d.z;
            a |= 1;
        }
        
        const double tx0 = (ray.d.x == 0) ? DBL_MIN : (min.x - ray.o.x) * (1. / ray.d.x);
        const double tx1 = (ray.d.x == 0) ? DBL_MAX : (max.x - ray.o.x) * (1. / ray.d.x);
        const double ty0 = (ray.d.y == 0) ? DBL_MIN : (min.y - ray.o.y) * (1. / ray.d.y);
        const double ty1 = (ray.d.y == 0) ? DBL_MAX : (max.y - ray.o.y) * (1. / ray.d.y);
        const double tz0 = (ray.d.z == 0) ? DBL_MIN : (min.z - ray.o.z) * (1. / ray.d.z);
        const double tz1 = (ray.d.z == 0) ? DBL_MAX : (max.z - ray.o.z) * (1. / ray.d.z);
        // printf("%lf, %lf, %lf, %lf, %lf, %lf\n", tx0, ty0, tz0, tx1, ty1, tz1);

        // const double tx0 = (min.x - ray.o.x) * (1. / ray.d.x);
        // const double tx1 = (max.x - ray.o.x) * (1. / ray.d.x);
        // const double ty0 = (min.y - ray.o.y) * (1. / ray.d.y);
        // const double ty1 = (max.y - ray.o.y) * (1. / ray.d.y);
        // const double tz0 = (min.z - ray.o.z) * (1. / ray.d.z);
        // const double tz1 = (max.z - ray.o.z) * (1. / ray.d.z);


        if(fmax(fmax(tx0, ty0), tz0) < fmin(fmin(tx1, ty1), tz1)) {
            int foundNode = -1;
            Stack stack;
            foundNode = traverseNewNode(tx0, ty0, tz0, tx1, ty1, tz1, 0, stack, nodes, ray);
            while (!stack.isEmpty() && foundNode == -1) {
                Stack::SubtreeData* data = stack.top();
                foundNode = traverseChildNodes(data, a, stack, nodes, ray);
            }
            return foundNode;
        }
        return -2;
    }

    __host__ __device__ int traverseChildNodes(Stack::SubtreeData* data, const unsigned char& a, Stack& stack, const Node* nodes, Ray& ray) {

        switch (data->nextQuadrant) {
        case 0:
            data->nextQuadrant = getNextQuadrant(data->txm, 4, data->tym, 2, data->tzm, 1);
            return traverseNewNode(data->tx0, data->ty0, data->tz0, data->txm, data->tym, data->tzm, nodes[data->nodeIdx].firstChildIdx + a, stack, nodes, ray);
        case 1:
            data->nextQuadrant = getNextQuadrant(data->txm, 5, data->tym, 3, data->tz1, 8);
            return traverseNewNode(data->tx0, data->ty0, data->tzm, data->txm, data->tym, data->tz1, nodes[data->nodeIdx].firstChildIdx + (1 ^ a), stack, nodes, ray);
        case 2:
            data->nextQuadrant = getNextQuadrant(data->txm, 6, data->ty1, 8, data->tzm, 3);
            return traverseNewNode(data->tx0, data->tym, data->tz0, data->txm, data->ty1, data->tzm, nodes[data->nodeIdx].firstChildIdx + (2 ^ a), stack, nodes, ray);
        case 3:
            data->nextQuadrant = getNextQuadrant(data->txm, 7, data->ty1, 8, data->tz1, 8);
            return traverseNewNode(data->tx0, data->tym, data->tzm, data->txm, data->ty1, data->tz1, nodes[data->nodeIdx].firstChildIdx + (3 ^ a), stack, nodes, ray);
        case 4:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->tym, 6, data->tzm, 5);
            return traverseNewNode(data->txm, data->ty0, data->tz0, data->tx1, data->tym, data->tzm, nodes[data->nodeIdx].firstChildIdx + (4 ^ a), stack, nodes, ray);
        case 5:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->tym, 7, data->tz1, 8);
            return traverseNewNode(data->txm, data->ty0, data->tzm, data->tx1, data->tym, data->tz1, nodes[data->nodeIdx].firstChildIdx + (5 ^ a), stack, nodes, ray);
        case 6:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->ty1, 8, data->tzm, 7);
            return traverseNewNode(data->txm, data->tym, data->tz0, data->tx1, data->ty1, data->tzm, nodes[data->nodeIdx].firstChildIdx + (6 ^ a), stack, nodes, ray);
        case 7:
            data->nextQuadrant = 8;
            return traverseNewNode(data->txm, data->tym, data->tzm, data->tx1, data->ty1, data->tz1, nodes[data->nodeIdx].firstChildIdx + (7 ^ a), stack, nodes, ray);
        case 8:
            stack.pop();
            return -1;
        }

        return -1;
    }

    __host__ __device__ int traverseNewNode(const float& tx0, const float& ty0, const float& tz0, const float& tx1, const float& ty1, const float& tz1, const int& nodeIdx, Stack& stack, const Node* nodes, Ray& ray) {
        
        if (tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) {
            // printf("%lf, %lf, %lf\n", tx1, ty1, tz1);
            return -1;
        }
        // printf("OH?\n");
        if (nodes[nodeIdx].isLeaf) {
            ray.value += 0.7;
            return -1;
        }

        const float txm = 0.5f * (tx0 + tx1);
        const float tym = 0.5f * (ty0 + ty1);
        const float tzm = 0.5f * (tz0 + tz1);
    
        stack.push({
            tx0, ty0, tz0,
            txm, tym, tzm,
            tx1, ty1, tz1,
            nodeIdx,
            getFirstQuadrant(tx0, ty0, tz0, txm, tym, tzm)
        });
        
        return -1;
    }

    __host__ __device__ int getFirstQuadrant(const float& tx0, const float& ty0, const float& tz0, const float& txm, const float& tym, const float& tzm) {
        unsigned char a = 0;

        if (tx0 > ty0) {
            if (tx0 > tz0){
                if (tym < tx0) a |= 2;
                if (tzm < tx0) a |= 1;
                return (int) a;
            }
        }

        else {
            if (ty0 > tz0) {
                if (txm < ty0) a |= 4;
                if (tzm < ty0) a |= 1;
                return (int) a;
            }
        }

        if (txm < tz0) a |= 4;
        if (tym < tz0) a |= 2;
        return (int) a;
    }

    __host__ __device__ int getNextQuadrant(const float& txm, const int& x, const float& tym, const int& y, const float& tzm, const int& z) {

        if (txm < tym) {
            if (txm < tzm) return x;
        }

        else {
            if (tym < tzm) return y;
        }
        
        return z;
    }

};

#endif // __OCTREE_H__