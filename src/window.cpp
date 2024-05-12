#include "graphic/window.h"
#include <algorithm>

#include <chrono>
#include <string>
#include <vector>

#include "loader/depth.h"
#include "loader/intrinsic.h"
#include "loader/pose.h"

void update(octomap::OcTree* tree, const Depth& depth, const Intrinsic& intrinsic, const Pose& pose) {

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
        tree->insertPointCloud(local, origin);
    }
}

void getOctreeVertices(const octomap::OcTree* tree, std::vector<GLfloat>& vertices, std::vector<GLfloat>& colors) {

    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(), end = tree->end_leafs(); it != end; ++it) {
        float size = (float)it.getSize();
        float x = (float)it.getX();
        float y = (float)it.getY();
        float z = (float)it.getZ();

        // 각 voxel의 vertices 계산
        GLfloat cubeVertices[] = {
            x - size / 2, y - size / 2, z + size / 2,  // Top-left
            x + size / 2, y - size / 2, z + size / 2,  // Top-right
            x + size / 2, y - size / 2, z - size / 2,  // Bottom-right
            x - size / 2, y - size / 2, z - size / 2,  // Bottom-left
            x - size / 2, y + size / 2, z + size / 2,  // Top-left
            x + size / 2, y + size / 2, z + size / 2,  // Top-right
            x + size / 2, y + size / 2, z - size / 2,  // Bottom-right
            x - size / 2, y + size / 2, z - size / 2   // Bottom-left
        };

        // 점유 확률에 따른 색상 계산
        float occupancy = it->getOccupancy();
        GLfloat color[] = {
            occupancy, 0.0f, 1.0f - occupancy, 1.0f  // RGBA
        };

        vertices.insert(vertices.end(), std::begin(cubeVertices), std::end(cubeVertices));
        colors.insert(colors.end(), std::begin(color), std::end(color));
    }

}

void Window::frameBufferSizeCallback(GLFWwindow* window, int width, int height){
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->reshape(width, height);
}

void Window::keycallback(GLFWwindow* window, int key ,int scancode, int action, int mods) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if(win) win->keyevent(key, scancode, action, mods);
}

void Window::keyevent(int key, int scancode, int action, int mods) {

    // float cameraSpeed = 0.05f;
    // if (key == GLFW_KEY_W && action == GLFW_PRESS) {
    //     glm::vec3 cameraDirection = glm::normalize(m_cameraPos - m_cameraTarget);
    //     m_cameraPos -= cameraSpeed * cameraDirection;
    //     m_cameraTarget -= cameraSpeed * cameraDirection;
    //     printf("Pos : %f, %f, %f \n", m_cameraPos.x, m_cameraPos.y, m_cameraPos.z);
    // }
    // if (key == GLFW_KEY_S && action == GLFW_PRESS) {
    //     glm::vec3 cameraDirection = glm::normalize(m_cameraPos - m_cameraTarget);
    //     m_cameraPos += cameraSpeed * cameraDirection;
    //     m_cameraTarget += cameraSpeed * cameraDirection;
    // }
    // if (key == GLFW_KEY_A && action == GLFW_PRESS) {
    //     glm::vec3 cameraDirection = glm::normalize(glm::cross(m_upVector, m_cameraPos - m_cameraTarget));
    //     m_cameraPos -= cameraSpeed * cameraDirection;
    //     m_cameraTarget -= cameraSpeed * cameraDirection;
    // }
    // if (key == GLFW_KEY_D && action == GLFW_PRESS) {
    //     glm::vec3 cameraDirection = glm::normalize(glm::cross(m_upVector, m_cameraPos - m_cameraTarget));
    //     m_cameraPos += cameraSpeed * cameraDirection;
    //     m_cameraTarget += cameraSpeed * cameraDirection;
    // }
    // m_view = glm::lookAt(m_cameraPos, m_cameraTarget, m_upVector);

    if(key == GLFW_KEY_U && action == GLFW_PRESS) {
        
        std::string datapath = "C:/DATASET/object-data/armadillo";
        Depth depth(datapath + "/depth/" + std::to_string(count) + ".npy");
        Intrinsic intrinsic(datapath + "/intrinsic/" + std::to_string(count) + ".txt");
        Pose pose(datapath + "/pose/" + std::to_string(count) + ".txt");
        update(m_tree.get(), depth, intrinsic, pose);
        count++;

        getOctreeVertices(m_tree.get(), vertices, colors);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, colorvbo);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

void Window::keytest() {
    float cameraSpeed = 0.05f;

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
    m_view = glm::lookAt(
        m_cameraPos,
        m_cameraPos + m_cameraFront,
        m_cameraUp);
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
    glEnable(GL_ALPHA_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glDepthFunc(GL_LEQUAL);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // glfwSwapInterval(0);

    m_cameraPos = glm::vec3(0.0f, 0.0f, 1.0f);
    m_cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    m_cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    m_view = glm::lookAt(
        m_cameraPos,
        m_cameraPos + m_cameraFront,
        m_cameraUp);

    m_tree = std::make_unique<octomap::OcTree>(0.01);


    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &colorvbo);
    glBindBuffer(GL_ARRAY_BUFFER, colorvbo);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return true;

}

void Window::draw() {

    m_program->use();
    glBindVertexArray(vao);

    m_program->setUniform("projection", m_projection);
    m_program->setUniform("view", m_view);

    glm::mat4 model(1.0f);

    m_program->setUniform("model", model);

    glDrawArrays(GL_POINTS, 0, vertices.size() / 3);
    glBindVertexArray(0);

}

void Window::render() {
    glfwPollEvents();
    glClearColor(0.1, 0.2, 0.3, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    keytest();
    draw();

    glfwSwapBuffers(m_window);
}

bool Window::loadShaderProgram() {

    std::shared_ptr<Shader> vertShader = Shader::createFromFile("shader/simple.vs", GL_VERTEX_SHADER);
    std::shared_ptr<Shader> fragShader = Shader::createFromFile("shader/simple.fs", GL_FRAGMENT_SHADER);
    m_program = Program::create({vertShader, fragShader});
    if(!m_program) return false;
    return true;
}

