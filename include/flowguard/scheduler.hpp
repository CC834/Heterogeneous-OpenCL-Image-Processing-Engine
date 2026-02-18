#pragma once

#include "flowguard/types.hpp"

namespace flowguard {

class TileScheduler {
 public:
  TileScheduler(ExecutionMode mode, float initial_gpu_ratio = 0.5F,
                std::size_t update_interval = 30);

  float gpu_ratio() const;
  void observe(std::size_t cpu_tiles, double cpu_ms, std::size_t gpu_tiles,
               double gpu_ms);

 private:
  ExecutionMode mode_;
  float ratio_;
  std::size_t update_interval_;
  std::size_t frames_{};
  double cpu_rate_{};
  double gpu_rate_{};
};

}  // namespace flowguard
