#include "graphic/window.h"

void Window::keyCameraMove() {
    float cameraSpeed = 0.01f;
    float cameraRotateSpeed = 0.05f;

    if(glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
        m_cameraPos += cameraSpeed * m_cameraFront;
    if(glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
        m_cameraPos -= cameraSpeed * m_cameraFront;
    auto cameraRight = glm::normalize(glm::cross(m_cameraUp, -m_cameraFront));

    if(glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
        m_cameraPos += cameraSpeed * cameraRight;
    if(glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
        m_cameraPos -= cameraSpeed * cameraRight;
    auto cameraUp = glm::normalize(glm::cross(-m_cameraFront, cameraRight));

    if(glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS)
        if(glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            m_cameraPos -= cameraSpeed * cameraUp;
        else
            m_cameraPos += cameraSpeed * cameraUp;

    if(glfwGetKey(m_window, GLFW_KEY_UP) == GLFW_PRESS) {
        m_cameraPitch += cameraRotateSpeed * 20;
    }
    if(glfwGetKey(m_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        m_cameraPitch -= cameraRotateSpeed * 20;
    }
    if(glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        m_cameraYaw -= cameraRotateSpeed * 20;
    }
    if(glfwGetKey(m_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        m_cameraYaw += cameraRotateSpeed * 20;
    }

    if(m_cameraYaw < 0.0f) m_cameraYaw += 360.0f;
    if(m_cameraYaw > 360.0f) m_cameraYaw -= 360.0f;
    if(m_cameraPitch > 89.0f) m_cameraPitch = 89.0f;
    if(m_cameraPitch < -89.0f) m_cameraPitch = -89.0f;

    m_cameraFront =     
        glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraYaw), glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraPitch), glm::vec3(1.0f, 0.0f, 0.0f)) *
        glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);

    m_view = glm::lookAt(
        m_cameraPos,
        m_cameraPos + m_cameraFront,
        m_cameraUp);
}

void Window::keycallback(GLFWwindow* window, int key ,int scancode, int action, int mods) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->keyevent(key, scancode, action, mods);
}

void Window::keyevent(int key, int scancode, int action, int mods) {

    if(key == GLFW_KEY_U && action == GLFW_PRESS) {
        
        std::string datapath = "C:/DATASET/object-data/xyzrgb_dragon";
        Depth depth(datapath + "/depth/" + std::to_string(count) + ".npy");
        Intrinsic intrinsic(datapath + "/intrinsic/" + std::to_string(count) + ".txt");
        Pose pose(datapath + "/pose/" + std::to_string(count) + ".txt");
        m_mytree->updateWithDepth(depth, intrinsic, pose);
        count++;
    }
}
