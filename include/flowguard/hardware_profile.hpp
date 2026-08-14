#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "flowguard/types.hpp"

namespace flowguard {

// A declared constraint model, not a claim about any named physical board.
struct HardwareProfile {
  std::string name;
  std::string description;
  int frame_width{640};
  int frame_height{360};
  double target_fps{30.0};
  int declared_cpu_cores{};
  int declared_memory_mb{};
  double camera_latency_ms{};
  double camera_jitter_ms{};
  double actuation_latency_ms{};
  double sensor_noise_sigma{};
  int motion_blur_pixels{1};
  bool gpu_available{true};
  double throttle_after_s{-1.0};
  double thermal_penalty_ms{};
  double gpu_failure_after_s{-1.0};
};

const std::vector<HardwareProfile>& hardware_profiles();
const HardwareProfile& find_hardware_profile(const std::string& name);

class HardwareEmulator {
 public:
  explicit HardwareEmulator(HardwareProfile profile, std::uint32_t seed = 42);

  cv::Mat capture(const cv::Mat& source, std::uint64_t frame_id, double simulation_time_s,
                  VirtualHardwareState& state);
  ControlCommand deliver(const ControlCommand& requested, double simulation_time_s,
                         VirtualHardwareState& state);
  void apply_thermal_penalty(double simulation_time_s, VirtualHardwareState& state) const;
  bool gpu_available(double simulation_time_s) const;
  const HardwareProfile& profile() const { return profile_; }

 private:
  void update_state(double simulation_time_s, VirtualHardwareState& state) const;

  HardwareProfile profile_;
  std::uint32_t seed_{};
  cv::Mat previous_capture_;
  std::deque<ControlCommand> command_queue_;
  ControlCommand applied_command_;
};

}  // namespace flowguard
