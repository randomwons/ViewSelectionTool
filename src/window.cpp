#include "graphic/window.h"
#include <algorithm>

#include <chrono>
#include <string>
#include <vector>


void Window::frameBufferSizeCallback(GLFWwindow* window, int width, int height){
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->reshape(width, height);
}

void Window::reshape(int width, int height){
    m_width = width;
    m_height = height;
    float aspectRatio = width / float(height);

    m_projection = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
    glViewport(0, 0, width, height);
}

std::unique_ptr<Window> Window::create(const int width, const int height, const char* title) {

    auto window = std::unique_ptr<Window>(new Window());
    if(!window->initialize(width, height, title)){
        return nullptr;
    }
    return std::move(window);

}

bool Window::initialize(const int width, const int height, const char* title) {

    if(!glfwInit()){
        printf("Failed to initialize glfw\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if(!m_window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return false;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL context version : '%s'\n", reinterpret_cast<const char*>(glVersion));

    glfwSetWindowUserPointer(m_window, this);
    frameBufferSizeCallback(m_window, width, height);
    glfwSetFramebufferSizeCallback(m_window, frameBufferSizeCallback);
    glfwSetKeyCallback(m_window, keycallback);

    if(!loadShaderProgram()){
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_ALPHA_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // 
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // glfwSwapInterval(0);

    m_view = glm::lookAt(
        m_cameraPos,
        m_cameraPos + m_cameraFront,
        m_cameraUp);

    m_mytree = std::make_unique<MyOcTree>(0.01, octomap::point3d(-0.64, -0.64, -0.64), octomap::point3d(0.64, 0.64, 0.64));

    return true;

}

bool Window::loadShaderProgram() {

    std::shared_ptr<Shader> vertShader = Shader::createFromFile("shader/simple.vs", GL_VERTEX_SHADER);
    std::shared_ptr<Shader> fragShader = Shader::createFromFile("shader/simple.fs", GL_FRAGMENT_SHADER);
    m_program = Program::create({vertShader, fragShader});
    if(!m_program) return false;
    return true;
}

