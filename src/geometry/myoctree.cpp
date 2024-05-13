#include "geometry/myoctree.h"

void MyOcTree::updateWithDepth(const Depth& depth, const Intrinsic& intrinsic, const Pose& pose) {

    octomap::point3d origin(
        -(pose.data[0] * pose.data[3] + pose.data[4] * pose.data[7] + pose.data[8] * pose.data[11]),
        -(pose.data[1] * pose.data[3] + pose.data[5] * pose.data[7] + pose.data[9] * pose.data[11]),
        -(pose.data[2] * pose.data[3] + pose.data[6] * pose.data[7] + pose.data[10] * pose.data[11])
    );
    #pragma omp parallel
    {
        octomap::Pointcloud local;
        #pragma omp for nowait
        for(int y = 0; y < depth.height(); y++){
            for(int x = 0; x < depth.width(); x++) {
                double d = depth.data[y * depth.width() + x];
                if(d == 0) continue;
                
                double dx = (x - intrinsic.cx()) / intrinsic.fx();
                double dy = (y - intrinsic.cy()) / intrinsic.fy();
                double dz = 1;

                octomap::point3d dir = octomap::point3d(
                    pose.data[0] * dx + pose.data[4] * dy + pose.data[8] * dz,
                    pose.data[1] * dx + pose.data[5] * dy + pose.data[9] * dz,
                    pose.data[2] * dx + pose.data[6] * dy + pose.data[10] * dz
                );
                dir.normalized();
                
                local.push_back(origin + dir * d);
            }
        }
        #pragma omp critical
        insertPointCloud(local, origin);
    }

    getAllNodeOccupancy();

}

void MyOcTree::getAllNodeOccupancy() {

    int index = 0;
    for(auto it = pclist.begin(); it != pclist.end(); ++it) {
        auto node = search((*it).x(), (*it).y(), (*it).z());
        if(node) occupancies[index] = node->getOccupancy();        
        index++;
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, colorvbo);
    glBufferData(GL_ARRAY_BUFFER, occupancies.size() * sizeof(float), occupancies.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void MyOcTree::draw(Program* program, const glm::mat4& projection, const glm::mat4& view) {

    program->use();
    glBindVertexArray(vao);

    program->setUniform("projection", projection);
    program->setUniform("view", view);
    program->setUniform("model", model);

    glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, pclist.size());
    glBindVertexArray(0);

}