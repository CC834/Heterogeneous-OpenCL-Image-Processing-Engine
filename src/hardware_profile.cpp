#include "flowguard/hardware_profile.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

#include <opencv2/imgproc.hpp>

namespace flowguard {
namespace {

constexpr double simulator_fps = 30.0;

double deterministic_jitter(std::uint64_t frame_id, std::uint32_t seed) {
  std::uint64_t value = frame_id ^ (static_cast<std::uint64_t>(seed) << 32U);
  value ^= value >> 33U;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33U;
  return static_cast<double>(value % 2001U) / 1000.0 - 1.0;
}

void delay(double milliseconds) {
  if (milliseconds <= 0.0) return;
  std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(milliseconds));
}

}  // namespace

const std::vector<HardwareProfile>& hardware_profiles() {
  static const std::vector<HardwareProfile> profiles = {
      {"desktop-native", "Measured host pipeline with no synthetic hardware delay", 640, 360,
       30.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1, true, -1.0, 0.0, -1.0},
      {"edge-balanced-sim", "Moderate onboard camera, compute, and actuator constraints", 480,
       270, 30.0, 4, 4096, 8.0, 2.0, 4.0, 1.5, 3, true, -1.0, 0.0, -1.0},
      {"edge-constrained-sim", "Lower-rate CPU-only profile with noisy blurred imagery", 320,
       180, 20.0, 2, 2048, 15.0, 4.0, 9.0, 3.0, 5, false, -1.0, 0.0, -1.0},
      {"thermal-throttle-sim", "Temperature ramp with a declared processing penalty", 480, 270,
       30.0, 4, 4096, 8.0, 2.0, 5.0, 1.5, 3, true, 2.0, 14.0, -1.0},
      {"gpu-failure-sim", "GPU becomes unavailable after two simulated seconds", 480, 270,
       30.0, 4, 4096, 8.0, 2.0, 5.0, 1.5, 3, true, -1.0, 0.0, 2.0},
  };
  return profiles;
}

const HardwareProfile& find_hardware_profile(const std::string& name) {
  const auto& profiles = hardware_profiles();
  const auto found = std::find_if(profiles.begin(), profiles.end(),
                                  [&](const HardwareProfile& profile) { return profile.name == name; });
  if (found == profiles.end()) throw std::invalid_argument("unknown hardware profile: " + name);
  return *found;
}

HardwareEmulator::HardwareEmulator(HardwareProfile profile, std::uint32_t seed)
    : profile_(std::move(profile)), seed_(seed) {
  if (profile_.frame_width <= 0 || profile_.frame_height <= 0 || profile_.target_fps <= 0.0 ||
      profile_.target_fps > simulator_fps || profile_.motion_blur_pixels < 1) {
    throw std::invalid_argument("hardware profile has invalid dimensions, rate, or blur");
  }
}

void HardwareEmulator::update_state(double simulation_time_s, VirtualHardwareState& state) const {
  state.profile = profile_.name;
  state.simulated = profile_.name != "desktop-native";
  state.frame_width = profile_.frame_width;
  state.frame_height = profile_.frame_height;
  state.target_fps = profile_.target_fps;
  state.declared_cpu_cores = profile_.declared_cpu_cores;
  state.declared_memory_mb = profile_.declared_memory_mb;
  state.gpu_available = gpu_available(simulation_time_s);
  state.deadline_ms = 1000.0 / profile_.target_fps;
  state.throttled = profile_.throttle_after_s >= 0.0 &&
                    simulation_time_s >= profile_.throttle_after_s;
  double modelled_temperature = 39.0 + simulation_time_s * 4.0;
  if (state.throttled) {
    modelled_temperature = 39.0 + profile_.throttle_after_s * 4.0 +
                           (simulation_time_s - profile_.throttle_after_s) * 8.0;
  }
  state.temperature_c = state.simulated ? std::min(86.0, modelled_temperature) : 0.0;
}

bool HardwareEmulator::gpu_available(double simulation_time_s) const {
  return profile_.gpu_available &&
         (profile_.gpu_failure_after_s < 0.0 || simulation_time_s < profile_.gpu_failure_after_s);
}

cv::Mat HardwareEmulator::capture(const cv::Mat& source, std::uint64_t frame_id,
                                  double simulation_time_s, VirtualHardwareState& state) {
  if (source.empty()) throw std::invalid_argument("cannot emulate an empty camera frame");
  update_state(simulation_time_s, state);
  if (!state.simulated) {
    state.camera_latency_ms = 0.0;
    state.frame_reused = false;
    return source.clone();
  }
  const auto started = std::chrono::steady_clock::now();
  const double sample_number = std::floor(simulation_time_s * profile_.target_fps + 1e-9);
  const double previous_sample = std::floor(std::max(0.0, simulation_time_s - 1.0 / simulator_fps) *
                                            profile_.target_fps + 1e-9);
  state.frame_reused = !previous_capture_.empty() && frame_id > 0 &&
                       sample_number == previous_sample;

  cv::Mat captured;
  if (state.frame_reused) {
    captured = previous_capture_.clone();
  } else {
    cv::resize(source, captured, {profile_.frame_width, profile_.frame_height}, 0.0, 0.0,
               cv::INTER_AREA);
    if (profile_.motion_blur_pixels > 1) {
      cv::blur(captured, captured, {profile_.motion_blur_pixels, 1});
    }
    if (profile_.sensor_noise_sigma > 0.0) {
      cv::Mat noise(captured.size(), CV_16SC3);
      cv::RNG rng(static_cast<std::uint64_t>(seed_) * 0x9e3779b9U + frame_id);
      rng.fill(noise, cv::RNG::NORMAL, 0.0, profile_.sensor_noise_sigma);
      cv::Mat widened;
      captured.convertTo(widened, CV_16SC3);
      cv::add(widened, noise, widened, cv::noArray(), CV_16SC3);
      widened.convertTo(captured, CV_8UC3);
    }
    previous_capture_ = captured.clone();
  }
  const double requested_delay = std::max(
      0.0, profile_.camera_latency_ms +
               profile_.camera_jitter_ms * deterministic_jitter(frame_id, seed_));
  delay(requested_delay);
  state.camera_latency_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return captured;
}

ControlCommand HardwareEmulator::deliver(const ControlCommand& requested,
                                         double simulation_time_s,
                                         VirtualHardwareState& state) {
  update_state(simulation_time_s, state);
  const auto started = std::chrono::steady_clock::now();
  command_queue_.push_back(requested);
  const auto delayed_frames = static_cast<std::size_t>(
      std::floor(profile_.actuation_latency_ms * profile_.target_fps / 1000.0));
  if (command_queue_.size() > delayed_frames) {
    applied_command_ = command_queue_.front();
    command_queue_.pop_front();
  }
  delay(profile_.actuation_latency_ms);
  state.actuation_latency_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  state.applied_command = applied_command_;
  return applied_command_;
}

void HardwareEmulator::apply_thermal_penalty(double simulation_time_s,
                                             VirtualHardwareState& state) const {
  update_state(simulation_time_s, state);
  const double penalty = state.throttled ? profile_.thermal_penalty_ms : 0.0;
  const auto started = std::chrono::steady_clock::now();
  delay(penalty);
  state.thermal_penalty_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

}  // namespace flowguard
