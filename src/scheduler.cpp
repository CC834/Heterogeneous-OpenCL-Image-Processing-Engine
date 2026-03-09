#include "flowguard/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flowguard {

TileScheduler::TileScheduler(ExecutionMode mode, float initial_gpu_ratio,
                             std::size_t update_interval)
    : mode_(mode),
      ratio_(mode == ExecutionMode::Cpu ? 0.0F
             : mode == ExecutionMode::Gpu ? 1.0F
                                          : initial_gpu_ratio),
      update_interval_(update_interval) {
  if (initial_gpu_ratio < 0.0F || initial_gpu_ratio > 1.0F) {
    throw std::invalid_argument("GPU ratio must be between 0 and 1");
  }
  if (update_interval == 0) {
    throw std::invalid_argument("scheduler update interval must be positive");
  }
  if (mode_ == ExecutionMode::Fixed || mode_ == ExecutionMode::Adaptive) {
    ratio_ = std::clamp(initial_gpu_ratio, 0.05F, 0.95F);
  }
}

float TileScheduler::gpu_ratio() const { return ratio_; }

void TileScheduler::observe(std::size_t cpu_tiles, double cpu_ms,
                            std::size_t gpu_tiles, double gpu_ms) {
  if (mode_ != ExecutionMode::Adaptive) return;

  constexpr double alpha = 0.25;
  if (cpu_tiles > 0 && cpu_ms > 0.0) {
    const double sample = static_cast<double>(cpu_tiles) / cpu_ms;
    cpu_rate_ = cpu_rate_ == 0.0 ? sample : alpha * sample + (1.0 - alpha) * cpu_rate_;
  }
  if (gpu_tiles > 0 && gpu_ms > 0.0) {
    const double sample = static_cast<double>(gpu_tiles) / gpu_ms;
    gpu_rate_ = gpu_rate_ == 0.0 ? sample : alpha * sample + (1.0 - alpha) * gpu_rate_;
  }

  ++frames_;
  if (frames_ % update_interval_ != 0 || cpu_rate_ <= 0.0 || gpu_rate_ <= 0.0) return;

  const float target = static_cast<float>(gpu_rate_ / (cpu_rate_ + gpu_rate_));
  if (std::abs(target - ratio_) < 0.03F) return;
  const float delta = std::clamp(target - ratio_, -0.10F, 0.10F);
  ratio_ = std::clamp(ratio_ + delta, 0.05F, 0.95F);
}

}  // namespace flowguard
