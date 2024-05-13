#include "graphic/window.h"

void Window::render() {
    glfwPollEvents();
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    keyCameraMove();
    m_mytree->draw(m_program.get(), m_projection, m_view);

    glfwSwapBuffers(m_window);
}


