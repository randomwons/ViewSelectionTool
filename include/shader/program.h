#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include "common.h"
#include "shader/shader.h"

class Program {
public:
    static std::unique_ptr<Program> create(const std::vector<std::shared_ptr<Shader>>& shaders);

    ~Program();
    uint32_t get() const { return m_program; }
    void use() const;

    void setUniform(const std::string& name, const glm::mat4& value) const;

private:
    Program() {}
    bool link(const std::vector<std::shared_ptr<Shader>>& shaders);
    uint32_t m_program { 0 };

    
};

#endif // __PROGRAM_H__
