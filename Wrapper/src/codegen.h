constexpr double step = 0.01;
std::ofstream fout("data/e-1"s + extension);
fout << std::fixed << std::setprecision(10);

data.reserve(1001);
for (int i = 0; i != 1001; ++i)
{
	fout << i * step << ' ' << std::exp(-i * step) << '\n';
	data.emplace_back(i * step, std::exp(-i * step));
}