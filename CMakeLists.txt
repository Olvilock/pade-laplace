cmake_minimum_required(VERSION 3.29)
project(Pade-Laplace
  VERSION 0.0.0
  LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
FILE(GLOB CUDA_sources "src/*.cu" "src/device/*.cu")

add_library(pade-laplace SHARED src/spline.cpp ${CUDA_sources})
target_include_directories(pade-laplace PUBLIC include PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(pade-laplace PRIVATE ${Boost_LIBRARIES})

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
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
    --relocatable-device-code=true >)
endforeach()
