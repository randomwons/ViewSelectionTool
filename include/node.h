#ifndef __NODE_H__
#define __NODE_H__

#include <cuda_runtime.h>

struct Node {
    Node() = default;
    Node(int parentIdx, int firstChildIdx, bool isLeaf=true) : parentIdx(parentIdx), firstChildIdx(firstChildIdx), isLeaf(isLeaf) {}

    int parentIdx;
    int firstChildIdx;
    bool isLeaf;


};

#endif // __NODE_H__