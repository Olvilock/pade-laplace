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
