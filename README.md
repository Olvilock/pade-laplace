## Building from Source

Ensure you have the required dependencies:

 * CMake >= 3.25
 * System C/C++ Toolchain
 * System CUDA Toolchain

Build commands:
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```

## Usage

Right now the program computes the Pade-Laplace approximmants in the hardcoded points, and prints the results. Sample usage:
```
build/pl-fit samples/e-1_1000.txt 32
```
