#include <plSolver.h>
#include <kernel/batchedSolver.cuh>

#include <device/transform.cuh>
#include <transformTypes.cuh>
#include <spline.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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

namespace pl
{
	DeviceNode::DeviceNode(const Node& node) :
		point{ node.point }, value{ node.value } {}

	using complex = thrust::complex<double>;

	int operator & (batchedStatus a, batchedStatus b)
	{
		return static_cast<int>(a) &
			   static_cast<int>(b);
	}

	template <>
	Approximation solveBatched<transformType::Trapezia> (
		const dataset_type& h_data, unsigned depth)
	{
		thrust::device_vector<DeviceNode> d_data = h_data;
		auto less_point =
			[] __device__ (const DeviceNode & a, const DeviceNode & b)
			{ return a.point < b.point; };
		thrust::sort(thrust::device, d_data.begin(), d_data.end(), less_point);
		
		std::vector<thrust::complex<double> > h_grid
		{
			{ 5.0, 0.0 },
			//{ 3.0, 0.0 },
			//{ 3.0, 10.0 }
		};
		thrust::device_vector<thrust::complex<double> > d_grid = h_grid;
		
		thrust::device_vector<batchedResult> d_result(
			d_grid.size() * depth * (depth + 1) / 2,
			{ {}, batchedStatus::untouched });
		
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaCheckErrors("Device error\n");
		std::cout << "Kernel launch...\n";
		
		kernelBatchedVecSolver
		<<< d_grid.size(), depth, 4 * depth * sizeof(complex) >>>
			(d_grid.data().get(), d_result.data().get(),
				TrapeziaData{ d_data.data().get(), (unsigned)d_data.size() });
		
		cudaCheckErrors("Kernel launch failed\n");
		
		cudaDeviceSynchronize();
		cudaCheckErrors("Could not synchronize\n");
		std::cout << "Kernel finished\n";
		
		thrust::host_vector<batchedResult> h_result = d_result;
		auto res_it = h_result.begin();
		for (auto s : h_grid)
		{
			std::cout << "For point p = " << s << " we have:\n";
			for (int count = 1; count <= depth; res_it += count++)
			{
				std::cout << "count = " << count << ":\n";
				auto cur_it = res_it;
				for (int id = 0; id < count; id++, cur_it++)
				{
					if ((*cur_it).status & batchedStatus::degenerate_system)
						std::cout << "(degenerate)";
					if ((*cur_it).status & batchedStatus::Aberth_divergence)
						std::cout << "(divergence)";
					if ((*cur_it).status & batchedStatus::untouched)
						std::cout << "(untouched)";
					if ((*cur_it).status == batchedStatus::ok)
						std::cout << "(ok)";
					
					auto data = (*cur_it).data;
					std::cout << "  a_" << id << " = " << data.amp <<
								 ", b_" << id << " = " << data.exp << "\n";
				}
			}
		}
		return {};
	}
}