cmake_minimum_required(VERSION 3.20)

project(glad)

# Include directory
include_directories(include)

# Source files
set(GLAD_SRC "src/gles2.c")

# Static library
add_library(glad STATIC ${GLAD_SRC})

# Set the location for library installation
install(TARGETS glad DESTINATION lib)
install(FILES include/glad/glad.h DESTINATION include/glad)