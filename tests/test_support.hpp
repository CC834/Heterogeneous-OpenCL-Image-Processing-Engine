#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace flowguard::test {

struct Case { std::string name; std::function<void()> body; };
std::vector<Case>& cases();

struct Register {
  Register(std::string name, std::function<void()> body);
};

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace flowguard::test

#define FLOWGUARD_TEST_JOIN_INNER(a, b) a##b
#define FLOWGUARD_TEST_JOIN(a, b) FLOWGUARD_TEST_JOIN_INNER(a, b)
#define TEST_CASE(name) \
  static void FLOWGUARD_TEST_JOIN(test_body_, __LINE__)(); \
  static ::flowguard::test::Register FLOWGUARD_TEST_JOIN(test_register_, __LINE__)( \
      name, FLOWGUARD_TEST_JOIN(test_body_, __LINE__)); \
  static void FLOWGUARD_TEST_JOIN(test_body_, __LINE__)()
