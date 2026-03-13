# CMake Tutorial:
#   https://cmake.org/cmake/help/v3.31/guide/tutorial/index.html
#   https://cmake.org/cmake/help/book/mastering-cmake/cmake/Help/guide/tutorial/index.html

cmake_minimum_required(VERSION 3.15)
# Starting from version 3.15  command `$ cmake --install .` is available, instead of `make install`

# Set the PROJECT_NAME and PROJECT_VERSION
project(tensorxx VERSION 1.0 LANGUAGES CXX)
# External usage:
#   find_package([PROJECT_NAME] CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE [NAMESPACE]::[target])
# Also, can be imported using:
#   add_subdirectory(Tensor-library)
#   target_link_libraries(app PUBLIC tensorxx)
# see IMPORT.md

# Static library target, the file will be named `lib[target].a` (see also Shared Library)
add_library(tensorxx STATIC src/Tensor.cpp)
set_target_properties(tensorxx PROPERTIES POSITION_INDEPENDENT_CODE ON) # -fPIC

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20) # 98
set(CMAKE_CXX_STANDARD_REQUIRED True)
message(STATUS "Compiler used: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")


# OpenCV
#   https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
#   CONFIG - find and import using:
#       opencv/cmake/OpenCVConfig.cmake.in -> /usr/local/lib/cmake/OpenCV/OpenCVConfig.cmake
#       (We implement same config mode approach: `tensorxx-config.cmake`).
find_package(OpenCV CONFIG REQUIRED)
target_include_directories(tensorxx PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tensorxx PUBLIC ${OpenCV_LIBS})


# option() without specifying [value] defaults to OFF, but here we explicitly specify ON
option(USE_EXAMPLES "Build example executables" ON)
if(USE_EXAMPLES)
    add_subdirectory(examples)
endif()


# Simple single source of truth
set(MY_CUSTOM_VARIABLE "MY CUSTOM VARIABLE")
# configure_file() must be after all set()/option() commands
configure_file(Config.h.in tensorxx/Config.h)
# Where to search for Config.h?
#   build and used locally:
target_include_directories(tensorxx PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
#   installed:
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tensorxx/Config.h" DESTINATION include/tensorxx)
# /usr/local/include/tensorxx/Config.h


# Include paths - add dirs to the list of paths to search for .h include files
target_include_directories(tensorxx
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>" # when library already installed
)
install(DIRECTORY include/ DESTINATION include)

# Install library
install(
    TARGETS tensorxx # targets,
    EXPORT tensorxx_targets # export set
    #
    LIBRARY DESTINATION lib/tensorxx # .so
    ARCHIVE DESTINATION lib/tensorxx # .a / .lib
    RUNTIME DESTINATION bin/tensorxx # .exe + .dll
    #
    INCLUDES DESTINATION include # same include paths with target_include_directories
)
# Build, install, and that's it - you can now use #include “tensorxx/Tensor.h” in your app


# Export, to find with find_package()
#   https://cmake.org/cmake/help/v3.31/command/find_package.html
#
# Config mode
install(
    EXPORT tensorxx_targets
    FILE tensorxx_targets.cmake # add_library(tensorxx::tensorxx...), add_library(...)
    NAMESPACE tensorxx::
    DESTINATION lib/cmake/tensorxx
)
#
# Create standard config file to be found and executed by find_package()
include(CMakePackageConfigHelpers)
# as configure_file(), generates the config file that includes the exports
configure_package_config_file(
    # in
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in # custom file, it will include(..tensorxx_targets.cmake)
    # out
    "${CMAKE_CURRENT_BINARY_DIR}/tensorxx-config.cmake"
    INSTALL_DESTINATION lib/cmake/tensorxx
    # NO_SET_AND_CHECK_MACRO
    # NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# additionally generate file with version
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensorxx-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion # AnyNewerVersion
)
# Locate/copy generated files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/tensorxx-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorxx-config-version.cmake"
    DESTINATION lib/cmake/tensorxx
)
#
# If you don't want to install
# Export targets to an outside project directly from a build tree with no installation involved
#   https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html#exporting-targets-from-the-build-tree
export(
    EXPORT tensorxx_targets # notice that it is defined even we didn't execute corresponding install commands
    FILE "${CMAKE_CURRENT_BINARY_DIR}/tensorxx_targets.cmake"
    NAMESPACE tensorxx::
)
