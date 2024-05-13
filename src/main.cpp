#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "graphic/window.h"

int main(){

    auto window = Window::create(1280, 720, "test");
    if(!window) {
        printf("Terminate program\n");
        return 0;
    }

    double lastTime = glfwGetTime();
    int frameCount = 0;
    while(!window->shouldClose()){
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastTime;
        frameCount++;
        if(deltaTime >= 1.0) {
            std::cout << "FPS : " << frameCount << ", Count : " << window->count << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
        window->render();

    }

    glfwTerminate();
    return 0;
}