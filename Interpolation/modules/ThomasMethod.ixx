export module Spline:ThomasMethod;

import <complex>;
import <array>;
import <vector>;

export namespace it
{
	std::vector<std::complex<double> > ThomasMethod
		(std::vector<std::array<std::complex<double>, 4> > coeff)
	{
		for (std::size_t i = 1; i < coeff.size(); ++i)
		{
			auto& [A, B, C, F] = coeff[i];
			const auto& [Ap, Bp, Cp, Fp] = coeff[i - 1];
			auto omega = A / Bp;
			B -= omega * Cp;
			F -= omega * Fp;
		}

		std::vector<std::complex<double> > result(coeff.size());

		auto coeff_it = coeff.rbegin();
		auto result_it = result.rbegin();
		{
			const auto& [A, B, C, F] = *coeff_it++;
			*result_it = F / B;
		}
		while (coeff_it != coeff.rend())
		{
			const auto& [A, B, C, F] = *coeff_it++;
			*result_it = (F - *result_it++ * C) / B;
		}

		return result;
	}
}
