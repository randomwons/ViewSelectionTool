#ifndef __COMMON_H__
#define __COMMON_H__

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// #include <spdlog/spdlog.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

std::optional<std::string> LoadTextFile(const std::string& filename);

#endif // __COMMON_H__