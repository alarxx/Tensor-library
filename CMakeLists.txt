cmake_minimum_required(VERSION 3.15)
# 3.15: $ cmake --install .

# Set the project name and version
project(TensorX VERSION 1.0 LANGUAGES CXX)

# Static library target named MathFunctions (see also Shared Library)
add_library(TensorX STATIC src/Tensor.cpp)
set_target_properties(TensorX PROPERTIES POSITION_INDEPENDENT_CODE ON) # -fPIC

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20) # 98
set(CMAKE_CXX_STANDARD_REQUIRED True)
message(STATUS "Compiler used: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(TensorX PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(TensorX PUBLIC ${OpenCV_LIBS})

# option() without specifying [value] defaults to OFF
option(USE_EXAMPLES "Build example executables" ON)
if(USE_EXAMPLES)
    add_subdirectory(examples)
endif()


# Single source of truth
set(MY_CUSTOM_VARIABLE "MY CUSTOM VARIABLE")
# configure_file() must be after all set()/option() commands
configure_file(Config.h.in TensorX/Config.h)
target_include_directories(TensorX PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TensorX/Config.h" DESTINATION include/TensorX)


# Include paths - add dirs to the list of paths to search for .h include files
target_include_directories(TensorX
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>" # when library already installed
)

# Install and Export, to find with find_package()
install(
    TARGETS TensorX
    EXPORT TensorXTargets
        LIBRARY DESTINATION lib/TensorX # .so
        ARCHIVE DESTINATION lib/TensorX # .a / .lib
        RUNTIME DESTINATION bin/TensorX # .exe + .dll
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

# Export

# Config mode
install(
    EXPORT TensorXTargets
    FILE TensorXTargets.cmake
    NAMESPACE TensorX::
    DESTINATION lib/cmake/TensorX
)

include(CMakePackageConfigHelpers)

# create config file
# as configure_file(), generates the config file that includes the exports
configure_package_config_file(
    # in
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    # out
    "${CMAKE_CURRENT_BINARY_DIR}/TensorXConfig.cmake"
    INSTALL_DESTINATION lib/cmake/TensorX
    # NO_SET_AND_CHECK_MACRO
    # NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# additionally generate file with version
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TensorXConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/TensorXConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TensorXConfigVersion.cmake"
    DESTINATION lib/cmake/TensorX
)


# export targets to an outside project directly from a build tree with no installation involved
export(
    EXPORT TensorXTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/TensorXTargets.cmake"
    NAMESPACE TensorX::
)
