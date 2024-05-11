#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "common.h"
#include "shader/shader.h"
#include "shader/program.h"

#include "geometry/cube.h"

class Window {
public:
    static std::unique_ptr<Window> create(const int height, const int width, const char* title);

    static void frameBufferSizeCallback(GLFWwindow* window, int width, int height);

    bool shouldClose() const {return glfwWindowShouldClose(m_window); }

    void render();

private:
    Window() {}
    void reshape(int height, int width);
    bool initialize(const int height, const int width, const char* title);
    bool loadShaderProgram();

    void sortObjectsByCameraDistance();

    std::shared_ptr<Program> m_program;
    const char* title;
    uint32_t m_height;
    uint32_t m_width;
    glm::mat4 m_projection;
    glm::mat4 m_view;

    GLFWwindow* m_window;

    std::shared_ptr<Cube> m_cubes;
    
};




#endif // __WINDOW_H__