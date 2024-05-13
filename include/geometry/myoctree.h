#ifndef __MYOCTREE_H__
#define __MYOCTREE_H__

#include "common.h"

#include "octomap/octomap.h"
#include "octomap/OcTree.h"

#include "loader/depth.h"
#include "loader/pose.h"
#include "loader/intrinsic.h"

#include "shader/program.h"

class MyOcTree : public octomap::OcTree {
public:
    MyOcTree(double resolution, octomap::point3d bbxmin, octomap::point3d bbxmax) : OcTree(resolution) {
        
        setBBXMin(bbxmin);
        setBBXMax(bbxmax);
        getUnknownLeafCenters(pclist, bbxmin, bbxmax);

        float vertices[] = {
            -0.005, -0.005,  0.005,
            -0.005,  0.005,  0.005,
             0.005,  0.005,  0.005,
             0.005, -0.005,  0.005,
            -0.005, -0.005, -0.005,
            -0.005,  0.005, -0.005,
             0.005,  0.005, -0.005,
             0.005, -0.005, -0.005,
        };

        uint32_t indices[] = {
            3, 0, 1,
            3, 1, 2,
            2, 1, 5,
            2, 5, 6,
            6, 5, 4,
            6, 4, 7,
            7, 4, 0,
            7, 0, 3,
            0, 4, 5,
            0, 5, 1,
            7, 3, 2,
            7, 2, 6
        };

        int index = 0;
        std::vector<glm::vec3> offsets(pclist.size());
        for (int z = 0; z < 127; ++z) {
            for (int y = 0; y < 127; ++y) {
                for (int x = 0; x < 127; ++x) {
                    offsets[index++] = glm::vec3(x * 0.01, y * 0.01, z * 0.01);
                }
            }
        }
        occupancies.resize(pclist.size(), 0.5);
        

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &colorvbo);
        glBindBuffer(GL_ARRAY_BUFFER, colorvbo);
        glBufferData(GL_ARRAY_BUFFER, occupancies.size() * sizeof(float), occupancies.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(GLfloat), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1, 1);

        glGenBuffers(1, &offsetvbo);
        glBindBuffer(GL_ARRAY_BUFFER, offsetvbo);
        glBufferData(GL_ARRAY_BUFFER, offsets.size() * sizeof(glm::vec3), offsets.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(2);
        glVertexAttribDivisor(2, 1);

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);


    }

    void updateWithDepth(const Depth& depth, const Intrinsic& intrinsic, const Pose& pose);
    void draw(Program* program, const glm::mat4& projection, const glm::mat4& view);

private:
    void getAllNodeOccupancy();

    glm::mat4 model { glm::mat4(1.0f) };

    uint32_t vao;
    uint32_t vbo;
    uint32_t ebo;
    uint32_t colorvbo;
    uint32_t offsetvbo;

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> colors;
    octomap::point3d_list pclist;
    std::vector<float> occupancies;

};



#endif // __MYOCTREE_H__