#include "flowguard/application.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    return flowguard::run_application(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "flowguard: " << error.what() << '\n';
    return 1;
  }
}
