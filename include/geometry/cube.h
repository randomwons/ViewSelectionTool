#ifndef __CUBE_H__
#define __CUBE_H__

#include "common.h"
#include "shader/program.h"

class Cube {
public:
    Cube();
    ~Cube();

    void addCube(const glm::mat4& model);
    void sortByCamera(const glm::vec3& cameraPos);
    uint32_t get() const { return m_vao; }
    void draw(std::shared_ptr<Program> program, 
              glm::mat4& projection,
              glm::mat4& view);

    std::vector<glm::vec3> m_positions;
    std::vector<float> distanceToCameras;

    size_t count = 0;

private:

    std::vector<glm::mat4> m_models;

    uint32_t m_vao { 0 };
    uint32_t m_vbo { 0 };
    uint32_t m_ebo { 0 };

    

};


#endif // __CUBE_H__