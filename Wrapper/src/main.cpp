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
	std::cout << std::fixed << std::setprecision(6);
	auto [result] = pl::solveBatched<pl::transformType::Trapezia>(data, 64);

	std::system("pause");
}