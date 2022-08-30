#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

#include <plSolver.h>
#include <spline.h>

int main(int argc, char* argv[])
{
	using namespace std::string_literals;
	namespace fs = std::filesystem;

	constexpr const char* extension = ".padelpltxt";
	
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " {infile}\n";
		return EXIT_FAILURE;
	}

	fs::path path = argv[1];
	std::cout << "Opening " << path << '\n';
	if (!exists(path))
	{
		std::cout << path << " does not exist!\n";
		return EXIT_FAILURE;
	}
	if (!is_regular_file(path) || path.extension() != extension)
	{
		std::cout << path << " is not a (suitable) file!\n";
		return EXIT_FAILURE;
	}

	pl::dataset_type data;
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
	auto spline = pl::getSpline(data);
	for (std::size_t id = 1; id != data.size(); ++id)
	{
		auto x = data[id - 1].Node - data[id].Node;
		auto& [a, b, c, d] = spline[id - 1];
		std::cout << a + x * (b + x * (c + x * d)) << ' ';
		std::cout << b + x * (2.0 * c + x * 3.0 * d) << ' ';
		std::cout << 2.0 * c + 6.0 * x * d << ' ' << 6.0 * d << '\n';
		std::cout << a << ' ' << b << ' ' << 2.0 * c << ' ' << 6.0 * d << '\n';
	}
	*/
	std::cout << std::fixed << std::setprecision(6);
	auto [result] = pl::solveBatched<pl::transformType::Trapezia>(data, 64);

	std::system("pause");
}