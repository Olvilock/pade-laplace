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

	constexpr const char* extension = ".padelpldata";
	
	pl::dataset_type data;
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " {infile}\n";
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

	std::cout << std::fixed << std::setprecision(10);
	auto [result] = pl::approx(data);

	std::system("pause");
}