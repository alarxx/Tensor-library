# 4 Ways to Import

## 1. add_subdirectory (local)

Mono-repository, Tensor-library is inside of app directory.

```python
cmake_minimum_required(VERSION 3.15)
project(app LANGUAGES CXX)

add_subdirectory(Tensor-library)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE tensorxx)
```

---

## 2. export (local)

In build directory:
- `build/tensorxx-config.cmake`
- `build/tensorxx-config-version.cmake`
- `build/tensorxx_targets.cmake`

```python
cmake_minimum_required(VERSION 3.15)
project(app LANGUAGES CXX)

add_executable(app main.cpp)

include("${CMAKE_CURRENT_SOURCE_DIR}/Tensor-library/build/tensorxx_targets.cmake")

target_link_libraries(app PRIVATE tensorxx::tensorxx)
```

```python
cmake_minimum_required(VERSION 3.15)
project(app LANGUAGES CXX)

add_executable(app main.cpp)

find_package(tensorxx CONFIG REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/Tensor-library/build"
    NO_DEFAULT_PATH
)

target_link_libraries(app PRIVATE tensorxx::tensorxx)
```

---

## 3. install (global)

Bad manual approach, but shows why `find_package()` is needed.

```python
cmake_minimum_required(VERSION 3.15)
project(app LANGUAGES CXX)

add_executable(app main.cpp)

target_include_directories(app PRIVATE /usr/local/include)
target_link_libraries(app PRIVATE /usr/local/lib/tensorxx/libtensorxx.a)
```

---

## 4. install and find_package (global)

`find_package()` uses 3 files:
- `tensorxx-config.cmake` - includes targets file
- `tensorxx_targets.cmake`- the targets file
- `tensorxx-config-version.cmake`

It searches in `/usr/local/lib/cmake/[package]/` files `<package-name>-config.cmake` or `<PackageName>Config.cmake` (see [link](https://cmake.org/cmake/help/v3.31/command/find_package.html#config-mode-search-procedure)).

```python
cmake_minimum_required(VERSION 3.15)
project(app LANGUAGES CXX)

find_package(tensorxx CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE tensorxx::tensorxx)
```

