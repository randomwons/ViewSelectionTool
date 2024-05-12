#include "graphic/window.h"
#include <algorithm>

#include <chrono>
#include <string>

#include "loader/depth.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"

void Window::frameBufferSizeCallback(GLFWwindow* window, int width, int height){
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->reshape(width, height);
}

void Window::keyCallbackWindow(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->keyCallback(key, scancode, action, mods);
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

void Window::keyCallback(int key, int scancode, int mods) {

    if(key == GLFW_KEY_U && action == GLFW_PRESS) {

        std::string datapath = "/home/vava/Dataset/armadillo";
        Depth depth(datapath + "/depth/" + std::to_string(0) + ".npy");
        Intrinsic intrinsic(datapath + "/intrinsic/" + std::to_string(0) + ".txt");
        Pose pose(datapath + "/pose/" + std::to_string(0) + ".txt");

        std::cout << intrinsic.fx() << std::endl;

    }

}


void Window::processInput() {


}

bool Window::initialize(const int width, const int height, const char* title) {

    if(!glfwInit()){
        printf("Failed to initialize glfw\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if(!m_window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);

    if(!gladLoadGLES2((GLADloadfunc)glfwGetProcAddress)){
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return false;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL context version : '%s'\n", reinterpret_cast<const char*>(glVersion));

    glfwSetWindowUserPointer(m_window, this);
    frameBufferSizeCallback(m_window, width, height);
    glfwSetFramebufferSizeCallback(m_window, frameBufferSizeCallback);

    if(!loadShaderProgram()){
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    // glEnable(GL_ALPHA_TEST);
    // glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // glDepthFunc(GL_LEQUAL);

    // glEnable(GL_CULL_FACE);
    // glCullFace(GL_BACK);

    glfwSwapInterval(0);

    m_tree = std::make_unique<octomap::OcTree>(0.01);


    glm::vec3 cameraPos = glm::vec3(3.0f, 5.0f, 15.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 upVector = glm::vec3(0.0f, 1.0f, 0.0f);
    m_view = glm::lookAt(cameraPos, cameraTarget, upVector);




    return true;

}

void Window::render() {
    glfwPollEvents();
    glClearColor(0.1, 0.2, 0.3, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glfwSwapBuffers(m_window);
}

bool Window::loadShaderProgram() {

    std::shared_ptr<Shader> vertShader = Shader::createFromFile("shader/simple.vs", GL_VERTEX_SHADER);
    std::shared_ptr<Shader> fragShader = Shader::createFromFile("shader/simple.fs", GL_FRAGMENT_SHADER);
    m_program = Program::create({vertShader, fragShader});
    if(!m_program) return false;
    return true;
}

