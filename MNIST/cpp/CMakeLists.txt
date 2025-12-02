cmake_minimum_required(VERSION 3.10)
project(mnist_purecpp)
set(CMAKE_CXX_STANDARD 17)
add_executable(mnist_train mnist_mlp.cpp)