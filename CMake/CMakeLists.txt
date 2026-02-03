cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_torch_project)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(my_torch_project main.cpp)

# Link against LibTorch
target_link_libraries(my_torch_project "${TORCH_LIBRARIES}")
set_property(TARGET my_torch_project PROPERTY CXX_STANDARD 17)
