import <iostream>;
import <fstream>;
import <filesystem>;
import <string>;

import Spline;
import <plApprox.h>;

int main(int argc, char* argv[])
{
	using namespace std::string_literals;
	namespace fs = std::filesystem;

	constexpr const char* extension = ".padelpltxt";
	
	pl::dataset_type data;
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " {infile}\n";
		
		constexpr double step = 0.1;
		std::ofstream fout("data/e-1_1000"s + extension);
		fout << std::fixed << std::setprecision(10);

		// data.reserve(101);
		for (int i = 0; i != 1001; ++i)
		{
			fout << i * step << ' ' << std::exp(-i * step) << '\n';
			// data.emplace_back(i * step, std::exp(-i * step));
		}
		
		return EXIT_FAILURE;
	}

	fs::path path = argv[1];
	if (!exists(path))
	{
		std::cout << argv[1] << " does not exist!\n";
		return EXIT_FAILURE;
	}
	if (!is_regular_file(path) || path.extension() != extension)
	{
		std::cout << argv[1] << " is not a (suitable) file!\n";
		return EXIT_FAILURE;
	}

	for (std::fstream fin(path);;)
	{
		double point, value;
		fin >> point >> value;
		if (!fin.good())
			break;
		data.emplace_back(point, value);
	}

	std::cout << "Dataset size is " << data.size() << '\n';
	/*
	auto spline = it::Spline(data).get_spline();
	for (std::size_t id = 1; id != data.size(); ++id)
	{
		auto x = data[id - 1].point - data[id].point;
		auto& [a, b, c, d] = spline[id - 1];
		std::cout << a + x * (b + x * (c + x * d)) << ' ';
		std::cout << b + x * (2.0 * c + x * 3.0 * d) << ' ';
		std::cout << 2.0 * c + 6.0 * x * d << ' ' << 6.0 * d << '\n';
		std::cout << a << ' ' << b << ' ' << 2.0 * c << ' ' << 6.0 * d << '\n';
	}
	*/
	std::cout << std::fixed << std::setprecision(6);
	auto [result] = pl::approx(data);

	std::system("pause");
}