#include "shader/program.h"

std::unique_ptr<Program> Program::create(const std::vector<std::shared_ptr<Shader>>& shaders) {
    auto program = std::unique_ptr<Program>(new Program());
    if(!program->link(shaders))
        return nullptr;
    return std::move(program);
}

bool Program::link(const std::vector<std::shared_ptr<Shader>>& shaders) {
    m_program = glCreateProgram();
    for(auto& shader : shaders)
        glAttachShader(m_program, shader->get());
    glLinkProgram(m_program);

    int success = 0;
    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetProgramInfoLog(m_program, 1024, nullptr, infoLog);
        printf("Failed to link program %s\n", infoLog);
        return false;
    }
    return true;
}

Program::~Program() {
    if(m_program) {
        glDeleteProgram(m_program);
    }
}

void Program::use() const {
    glUseProgram(m_program);
}

// void Program::SetUniform(const std::string& name, int value) const {
//     auto loc = glGetUniformLocation(m_program, name.c_str());
//     glUniform1i(loc, value);
// }

// void Program::SetUniform(const std::string& name, float value) const {
//     auto loc = glGetUniformLocation(m_program, name.c_str());
//     glUniform1f(loc, value);
// }

// void Program::SetUniform(const std::string& name, const glm::vec3& value) const {
//     auto loc = glGetUniformLocation(m_program, name.c_str());
//     glUniform3fv(loc, 1, glm::value_ptr(value));
// }

void Program::setUniform(const std::string& name, const glm::mat4& value) const {
    auto loc = glGetUniformLocation(m_program, name.c_str());
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(value));
}