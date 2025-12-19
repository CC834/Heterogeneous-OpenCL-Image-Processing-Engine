#include "test_support.hpp"

#include <iostream>

namespace flowguard::test {

std::vector<Case>& cases() {
  static std::vector<Case> value;
  return value;
}

Register::Register(std::string name, std::function<void()> body) {
  cases().push_back({std::move(name), std::move(body)});
}

}  // namespace flowguard::test

int main() {
  int failures = 0;
  for (const auto& test : flowguard::test::cases()) {
    try {
      test.body();
      std::cout << "PASS " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
    }
  }
  return failures == 0 ? 0 : 1;
}
