#include <solveAberth.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>

namespace solve
{
	namespace
	{
		using complex = thrust::complex<double>;

		__global__ void kernelAberth(const complex* const* polies, complex* const* roots,
									 double abs, double eps, double rot);
	}

	void Aberth(const thrust::device_vector<const complex*>& polies, unsigned degree,
				const thrust::device_vector<complex*>& roots,
				double absolute, double delta, double rotation)
	{
		assert(polies.size() == roots.size());
		kernelAberth <<< roots.size(), degree, (degree * 2 + 1) * sizeof(complex) >>>
			(polies.data().get(), roots.data().get(), absolute, delta, rotation);
	}

	namespace
	{
		struct Eval
		{
			complex value;
			complex prime;
		};

		__device__
		Eval evaluate(complex point)
		{
			extern __shared__ complex lc_poly[];

			complex exp_val = { 1.0 }, value = point;
			for (int exp = 1; exp < blockDim.x; exp <<= 1)
			{
				if (exp & threadIdx.x)
					exp_val *= value;
				value *= value;
			}

			value = {};
			complex prime{};
			for (int id = threadIdx.x + 1; ; ++id)
			{
				prime += id * lc_poly[id] * exp_val;

				exp_val *= point;
				value += lc_poly[id] * exp_val;

				if (id == blockDim.x)
				{
					id = 0;
					exp_val = { 1.0 };
					value += lc_poly[0];
				}
				if (id == threadIdx.x)
					break;
			}

			return { value, prime };
		}

		__global__ void kernelAberth(const complex* const* polies, complex* const* roots,
									 double abs, double eps, double rot)
		{
			const complex* glb_poly = polies[blockIdx.x];
			complex* glb_roots = roots[blockIdx.x];

			extern __shared__ complex lc_poly[];
			lc_poly[threadIdx.x] = glb_poly[threadIdx.x];
			if (!threadIdx.x)
				lc_poly[blockDim.x] = glb_poly[blockDim.x];

			complex* lc_roots = lc_poly + blockDim.x + 1;
			complex root = glb_roots[threadIdx.x];

			for (int it = 0; it != 100; ++it)
			{
				Eval eval = evaluate(root);
				if (thrust::abs(eval.value) < min(abs, eps * thrust::abs(eval.prime)))
				{
					glb_roots[threadIdx.x] = root;
					return;
				}

				lc_roots[threadIdx.x] = root;
				__syncthreads();

				complex delta = eval.prime / eval.value;
				for (int id = threadIdx.x + 1; ; ++id)
				{
					if (id == blockDim.x)
						id = 0;
					if (id == threadIdx.x)
						break;

					delta += 1.0 / (lc_roots[id] - root);
				}
				root -= complex{ 1.0, eps } / delta;
			}

			printf("Aberth: fallthrough (divergence) at thread %i\n",
				threadIdx.x + blockDim.x * blockIdx.x);
			__trap();
		}
	}
}