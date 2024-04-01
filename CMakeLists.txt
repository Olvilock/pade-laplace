cmake_minimum_required(VERSION 3.29)
project(Pade-Laplace
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FILE(GLOB CUDA_sources "src/*.cu" "src/device/*.cu")
add_library(pade-laplace src/spline.cpp ${CUDA_sources})

add_executable(pl-fit src/main.cpp)
target_link_libraries(pl-fit PRIVATE pade-laplace)

foreach(target pl-fit pade-laplace)
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "61"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION TRUE)
  target_include_directories(${target} PUBLIC include)
  target_compile_features(${target} PUBLIC cxx_std_20 cuda_std_20)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --relocatable-device-code=true
    >)
endforeach()
