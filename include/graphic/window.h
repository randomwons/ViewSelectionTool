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
    static void keycallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    bool shouldClose() const {return glfwWindowShouldClose(m_window); }

    void render();

private:
    Window() {}
    void reshape(int height, int width);
    void keyevent(int key, int scancode, int action, int mods);
    void keytest();
    bool initialize(const int height, const int width, const char* title);
    bool loadShaderProgram();

    void draw();

    std::shared_ptr<Program> m_program;
    const char* title;
    uint32_t m_height;
    uint32_t m_width;
    glm::mat4 m_projection;

    glm::vec3 m_cameraPos { glm::vec3(0.0f, 0.0f, 1.0f) };
    glm::vec3 m_cameraFront { glm::vec3(0.0f, 0.0f, -1.0f) };
    glm::vec3 m_cameraUp { glm::vec3(0.0f, 1.0f, 0.0f) };
    glm::mat4 m_view;

    GLFWwindow* m_window;
    
    std::unique_ptr<octomap::OcTree> m_tree;

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> colors;

    uint32_t vao { 0 };
    uint32_t vbo { 0 };
    uint32_t colorvbo { 0 };

    size_t count = 0;

};




#endif // __WINDOW_H__