cmake_minimum_required(VERSION 3.15)
# 3.15: $ cmake --install .

# Set the project name and version
project(ImportExample VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20) # 98
set(CMAKE_CXX_STANDARD_REQUIRED True)
message(STATUS "Compiler used: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Config mode: CONFIG
find_package(TensorX REQUIRED
    PATHS "${CMAKE_SOURCE_DIR}/TensorX/build"
)

# Add the main executable
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE TensorX::TensorX)

