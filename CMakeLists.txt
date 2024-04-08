cmake_minimum_required(VERSION 3.29)
project(Pade-Laplace
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
FILE(GLOB CUDA_sources "src/*.cu" "src/device/*.cu")

add_library(pade-laplace src/spline.cpp ${CUDA_sources})
target_include_directories(pade-laplace PUBLIC include)
target_compile_options(pade-laplace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --extended-lambda
  --relocatable-device-code=true
  >)

add_executable(pl-fit src/main.cpp)
target_include_directories(pl-fit PRIVATE include)
target_link_libraries(pl-fit PUBLIC pade-laplace)

foreach(target pl-fit pade-laplace)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 23
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 23
    CUDA_ARCHITECTURES "61"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION TRUE)
endforeach()
