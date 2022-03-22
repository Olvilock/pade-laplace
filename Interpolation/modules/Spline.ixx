export module Spline;

export import <cstddef>;
export import <vector>;
export import <array>;

export import <Point.h>;
export import <Spline.h>;

import :ThomasMethod;
import <algorithm>;

export namespace it
{
	Spline::Spline(const std::vector<Point>& data) :
		m_vertices(data.size() - 1), m_spline(data.size() - 1)
	{
		std::vector<std::array<double, 4> > forThomas(data.size());
		forThomas.front()[1] = forThomas.back()[1] = 1.0;

		{
			auto pred_diff = data[1].point - data[0].point;
			for (std::size_t i = 1; i + 1 < forThomas.size(); ++i)
			{
				auto cur_diff = data[i + 1].point - data[i].point;
				forThomas[i] =
				{
					pred_diff,
					2 * (cur_diff + pred_diff),
					cur_diff,
					3 * ((data[i + 1].value - data[i].value) / cur_diff -
							(data[i].value - data[i - 1].value) / pred_diff)
				};
				pred_diff = cur_diff;
			}
		}

		auto squareCoeff = ThomasMethod(std::move(forThomas));
		for (std::size_t i = 1; i < data.size(); ++i)
		{
			auto adj_diff = data[i].point - data[i - 1].point;
			m_vertices[i - 1] = data[i].point;
			m_spline[i - 1] =
			{
				data[i].value,
				(data[i].value - data[i - 1].value) / adj_diff +
					(2 * squareCoeff[i] + squareCoeff[i - 1]) * adj_diff / 3,
				squareCoeff[i],
				(squareCoeff[i] - squareCoeff[i - 1]) / (3 * adj_diff)
			};
		}
	}

	double Spline::operator () (double point) const
	{
		if (point > m_vertices.back())
			return {};
		auto it = std::lower_bound(m_vertices.cbegin(), m_vertices.cend(), point);
		if (it == m_vertices.cend())
			return {};

		auto delta = point - *it;
		const auto& [a, b, c, d] = m_spline[it - m_vertices.begin()];

		return a + delta * (b + delta * (c + delta * d));
	}

	const Spline::storage_type& Spline::get_spline() const&
	{
		return m_spline;
	}
	Spline::storage_type&& Spline::get_spline() &&
	{
		return std::move(m_spline);
	}
}
