#pragma once

#include "spline.h"

#include <complex>
#include <vector>

namespace pl {
struct Term {
  std::complex<double> mag;
  std::complex<double> exp;
};
struct Multiexp {
  std::vector<Term> terms;
};

enum class Method {
  Trapezia,
  Spline,
};

template <Method> Multiexp fit(const dataset_type &data, unsigned depth);
} // namespace pl
