#include <solveAberth.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

int main()
{
	std::vector<thrust::complex<double> > poly
	{
		{ -36.0, 0.0 },
		{ 0.0, 0.0 },
		{ 5.0, 0.0 },
		{ 0.0, 0.0 },
		{ 1.0, 0.0 },
	};

	std::vector<thrust::complex<double> > roots
	{
		{ 0.1, 0.0 },
		{ 0.0, 0.1 },
		{ -0.1, 0.0 },
		{ 0.0, -0.1 }
	};

	thrust::device_vector<thrust::complex<double> > d_roots = roots;

	std::cout << "Solution started\n";
	solve::Aberth(poly, roots.size(), d_roots);
	cudaDeviceSynchronize();
	cudaCheckErrors("Kernel failed\n");

	std::cout << "Solutions found:\n\n";

	std::cout << std::fixed;
	std::cout.precision(10);
	thrust::host_vector<thrust::complex<double> > h_roots = d_roots;
	for (auto root : h_roots)
		std::cout << root.real() << ' ' << root.imag() << '\n';
}