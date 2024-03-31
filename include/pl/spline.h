#pragma once

#include <complex>
#include <vector>

namespace pl {
struct Node {
  double point;
  std::complex<double> value;
};

struct Cubic {
  std::complex<double> a, b, c, d;
};

using dataset_type = std::vector<Node>;
using spline_type = std::vector<Cubic>;

spline_type getSpline(const dataset_type &);
} // namespace pl
