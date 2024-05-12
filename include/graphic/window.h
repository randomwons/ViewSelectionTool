#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "common.h"
#include "shader/shader.h"
#include "shader/program.h"

#include "octomap/octomap.h"
#include "octomap/OcTree.h"

class Window {
public:
    static std::unique_ptr<Window> create(const int height, const int width, const char* title);
    
    static void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallbackWindow(GLFWwindow* window, int key, int scancode, int action, int mods);


    bool shouldClose() const {return glfwWindowShouldClose(m_window); }

    void render();

private:
    Window() {}
    void processInput();
    void keyCallback(int key, int scancode, int action, int mods);
    void reshape(int height, int width);
    bool initialize(const int height, const int width, const char* title);
    bool loadShaderProgram();

    std::shared_ptr<Program> m_program;
    const char* title;
    uint32_t m_height;
    uint32_t m_width;
    glm::mat4 m_projection;
    glm::mat4 m_view;

    GLFWwindow* m_window;
    
    std::unique_ptr<octomap::OcTree> m_tree;

    uint32_t vao;
    uint32_t vbo;

    std::vector<GLfloat> vertices;

};




#endif // __WINDOW_H__