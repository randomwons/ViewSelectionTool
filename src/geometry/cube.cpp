#include "geometry/cube.h"

#include <algorithm>

Cube::Cube() {

    float vertices[] = {
        -0.1, -0.1,  0.1,
        -0.1,  0.1,  0.1,
         0.1,  0.1,  0.1,
         0.1, -0.1,  0.1,
        -0.1, -0.1, -0.1,
        -0.1,  0.1, -0.1,
         0.1,  0.1, -0.1,
         0.1, -0.1, -0.1
    };

    unsigned int indices[] = {
        // Front face
        0, 1, 2,
        1, 3, 2,
        // Back face
        7, 6, 4,
        6, 5, 4,
        // Left face
        3, 2, 7,
        2, 6, 7,
        // Right face
        4, 5, 0,
        5, 1, 0,
        // Top face
        1, 5, 2,
        5, 6, 2,
        // Bottom face
        4, 0, 7,
        0, 3, 7
    };
    
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 24, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);

    glGenBuffers(1, &m_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * 36, indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


}

Cube::~Cube(){
    if(m_vbo) glDeleteBuffers(1, &m_vbo);
    if(m_ebo) glDeleteBuffers(1, &m_ebo);
    if(m_vao) glDeleteVertexArrays(1, &m_vao);
}

void Cube::addCube(const glm::mat4& model) {
    m_models.emplace_back(model);
    m_positions.push_back(glm::vec3(model[3]));
    count++;
}

void Cube::sortByCamera(const glm::vec3& cameraPos) {

    // distanceToCameras.resize(count);
    // for(int i = 0; i < count; i++){
    //     distanceToCameras[i] = glm::length(cameraPos - m_positions[i]);
    // }

    std::sort(m_models.begin(), m_models.end(), [&cameraPos](const glm::mat4& a, const glm::mat4& b) {
        return glm::length(glm::vec3(a[3]) - cameraPos) > glm::length(glm::vec3(b[3]) - cameraPos);
    });

}

void Cube::draw(std::shared_ptr<Program> program, 
    glm::mat4& projection,
    glm::mat4& view) {
    program->use();
    program->setUniform("projection", projection);
    program->setUniform("view", view);

    glBindVertexArray(m_vao);
    for(auto& model : m_models) {
        program->setUniform("model", model);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}