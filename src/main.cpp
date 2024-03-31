#include <filesystem>
#include <fstream>
#include <iostream>

#include <pl/fit.h>

using namespace std::string_literals;
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  constexpr const char *extension = ".txt";
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " {infile}\n";
    return EXIT_FAILURE;
  }

  fs::path path = argv[1];
  std::cout << "Opening " << path << '\n';
  if (!exists(path)) {
    std::cout << path << " does not exist!\n";
    return EXIT_FAILURE;
  }
  if (!is_regular_file(path) || path.extension() != extension) {
    std::cout << path << " is not a (suitable) file!\n";
    return EXIT_FAILURE;
  }

  pl::dataset_type data;
  for (std::fstream fin(path);;) {
    double point, value;
    fin >> point >> value;
    if (!fin.good())
      break;
    data.emplace_back(point, value);
  }

  std::cout << "Dataset size is " << data.size() << '\n';
  std::cout << std::fixed << std::setprecision(6);
  pl::fit<pl::Method::Trapezia>(data, 16);
}
