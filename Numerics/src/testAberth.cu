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
	using complex = thrust::complex<double>;

	std::vector<complex> poly
	{
		{ 4.0, 0.0 },
		{ 0.0, 0.0 },
		{ -5.0, 0.0 },
		{ 0.0, 0.0 },
		{ 1.0, 0.0 },
	};

	std::vector<complex> roots
	{
		{ 1.0, 0.0 },
		{ 0.0, 1.0 },
		{ -1.0, 0.0 },
		{ 0.0, -1.0 },
	};

	thrust::device_vector<complex> d_roots = roots;
	thrust::device_vector<complex> d_poly = poly;

	thrust::device_vector<const complex*> c_polies(1, d_poly.data().get());
	thrust::device_vector<complex*> c_roots(1, d_roots.data().get());

	std::cout << "Solution started\n";
	solve::Aberth(c_polies, roots.size(), c_roots);
	cudaDeviceSynchronize();
	cudaCheckErrors("Kernel failed\n");

	std::cout << "Solutions found:\n\n";

	std::cout << std::fixed;
	std::cout.precision(10);
	thrust::host_vector<thrust::complex<double> > h_roots = d_roots;
	for (auto root : h_roots)
		std::cout << root.real() << ' ' << root.imag() << '\n';
}