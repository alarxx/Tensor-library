
# Add the main executable named Tutorial
add_executable(example1 example1.cpp)
add_executable(example2 example2.cpp)
add_executable(example3 example3.cpp)

# "" is a list, e.g. "one;two;three"
set(EXTRA_LIBS "")
set(EXTRA_INCLUDES "")
list(APPEND EXTRA_LIBS TensorX)
# list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")

# Transitive property
# PUBLIC:       MyApp -> Tutorial --->  MathFunctions
# PRIVATE:      MyApp -> Tutorial -X->  MathFunctions
# INTERFACE:    MyApp -> Tutorial --->  MathFunctions (but Tutorial doesn't use MathFunctions)

# Dependency links of other targets, be it static.a or shared.so linking libraries
target_link_libraries(example1 PRIVATE ${EXTRA_LIBS})
target_link_libraries(example2 PRIVATE ${EXTRA_LIBS})
target_link_libraries(example3 PRIVATE ${EXTRA_LIBS})
# vs. target_link_directories() - where to search for .a/.so (not needed, in general)


# # Install
# install(TARGETS Tutorial DESTINATION bin/Tutorial)
# install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h" DESTINATION include/Tutorial)
