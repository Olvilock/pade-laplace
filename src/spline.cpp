#include <pl/spline.h>

#include <array>
#include <complex>
#include <vector>

namespace pl {
namespace {
auto ThomasMethod(std::vector<std::array<std::complex<double>, 4>> coeff) {
  for (std::size_t i = 1; i < coeff.size(); ++i) {
    auto omega = coeff[i][0] / coeff[i - 1][1];
    coeff[i][1] -= omega * coeff[i - 1][2];
    coeff[i][3] -= omega * coeff[i][3];
  }

  std::vector<std::complex<double>> result(coeff.size());

  auto coeff_it = coeff.rbegin();
  auto result_it = result.rbegin();

  *result_it = (*coeff_it)[3] / (*coeff_it)[1];

  while (++coeff_it != coeff.rend())
    *result_it =
        ((*coeff_it)[3] - *result_it++ * (*coeff_it)[2]) / (*coeff_it)[1];

  return result;
}
} // namespace

spline_type getSpline(const pl::dataset_type &data) {
  std::vector<std::array<std::complex<double>, 4>> forThomas(data.size());
  forThomas.front()[1] = forThomas.back()[1] = 1.0;

  {
    auto pred_diff = data[1].point - data[0].point;
    for (std::size_t i = 1; i + 1 < forThomas.size(); ++i) {
      auto cur_diff = data[i + 1].point - data[i].point;
      forThomas[i] = {pred_diff, 2.0 * (cur_diff + pred_diff), cur_diff,
                      3.0 * ((data[i + 1].value - data[i].value) / cur_diff -
                             (data[i].value - data[i - 1].value) / pred_diff)};
      pred_diff = cur_diff;
    }
  }

  auto squareCoeff = ThomasMethod(std::move(forThomas));

  spline_type spline;
  spline.reserve(data.size() - 1);
  for (std::size_t i = 1; i < data.size(); ++i) {
    auto adj_diff = data[i].point - data[i - 1].point;
    spline.emplace_back(
        data[i].value,
        (data[i].value - data[i - 1].value) / adj_diff +
            (2.0 * squareCoeff[i] + squareCoeff[i - 1]) * adj_diff / 3.0,
        squareCoeff[i],
        (squareCoeff[i] - squareCoeff[i - 1]) / (3.0 * adj_diff));
  }

  return spline;
}
} // namespace pl
