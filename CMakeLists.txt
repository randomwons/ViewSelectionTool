cmake_minimum_required(VERSION 3.20)
project(Projective_Policy)

set(CMAKE_CXX_STANDARD 20)

find_package(octomap REQUIRED)
# find_package(OpenMP REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(dependencies/glad)
include_directories(include dependencies/glad/include)
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/window.cpp
    src/common.cpp
    src/shader/shader.cpp
    src/shader/program.cpp
)

# target_link_libraries(${PROJECT_NAME} ${OCTOMAP_LIBRARIES} OpenMP::OpenMP_CXX glad glfw)
target_link_libraries(${PROJECT_NAME} ${OCTOMAP_LIBRARIES} glad glfw)
target_include_directories(${PROJECT_NAME} PUBLIC ${OCTOMAP_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})

# Option to enable timer
option(USE_TIMER "Use the timer to measure execution time." ON)

if(USE_TIMER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TIMER)
endif()
