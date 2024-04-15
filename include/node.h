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

inline void buildNode(std::vector<Node>& nodes) {
    nodes.push_back(Node(-1, -1, true));
    int foo1 = splitNode(nodes, 0);
    for(int a = 0; a < 8; a++){
        int foo2 = splitNode(nodes, foo1 + a);
        for(int b = 0; b < 8; b++){
            int foo3 = splitNode(nodes, foo2 + b);
            for(int c = 0; c < 8; c++){
                int foo4 = splitNode(nodes, foo3 + c);
                for(int d = 0; d < 8; d++){
                    int foo5 = splitNode(nodes, foo4 + d);
                    for(int e = 0; e < 8; e++){
                        int foo6 = splitNode(nodes, foo5 + e);
                        for(int f = 0; f < 8; f++){
                            int foo7 = splitNode(nodes, foo6 + f);
                            // for(int g = 0; g < 8; g++){
                            //     int foo8 = splitNode(nodes, foo7 + g);
                            //     for(int h = 0; h < 8; h++){
                            //         int foo9 = splitNode(nodes, foo8 + h);
                            //     }
                            // }
                        }
                    }
                }
            }
        }
    } 
}

#endif // __NODE_H__