#include "test_support.hpp"

#include <cmath>

#include "flowguard/scheduler.hpp"

using flowguard::ExecutionMode;
using flowguard::TileScheduler;
using flowguard::test::require;

TEST_CASE("execution modes pin or clamp allocation") {
  require(TileScheduler(ExecutionMode::Cpu).gpu_ratio() == 0.0F, "CPU must receive every tile");
  require(TileScheduler(ExecutionMode::Gpu).gpu_ratio() == 1.0F, "GPU must receive every tile");
  require(std::abs(TileScheduler(ExecutionMode::Fixed, 0.7F).gpu_ratio() - 0.7F) < 1e-6F,
          "fixed ratio changed");
  require(std::abs(TileScheduler(ExecutionMode::Adaptive, 0.0F).gpu_ratio() - 0.05F) < 1e-6F,
          "adaptive lower bound failed");
}

TEST_CASE("adaptive scheduler observes interval hysteresis and step bounds") {
  TileScheduler scheduler(ExecutionMode::Adaptive, 0.5F, 30);
  for (int i = 0; i < 29; ++i) scheduler.observe(50, 100.0, 50, 10.0);
  require(std::abs(scheduler.gpu_ratio() - 0.5F) < 1e-6F, "scheduler updated before 30 frames");
  scheduler.observe(50, 100.0, 50, 10.0);
  require(std::abs(scheduler.gpu_ratio() - 0.6F) < 1e-5F, "scheduler did not enforce 10-point cap");
}
