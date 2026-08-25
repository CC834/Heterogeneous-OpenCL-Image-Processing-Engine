#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace flowguard {

enum class ExecutionMode { Cpu, Gpu, Fixed, Adaptive };
enum class WarningLevel { Clear, Yellow, Red };

struct Pose {
  double x{};
  double y{};
  double z{2.0};
  double yaw{};
};

struct GroundTruth {
  double nearest_obstacle_m{std::numeric_limits<double>::quiet_NaN()};
  double time_to_contact_s{std::numeric_limits<double>::quiet_NaN()};
  bool collision{};
};

struct SimulationFrame {
  std::uint64_t frame_id{};
  double simulation_time_s{};
  std::uint32_t seed{};
  cv::Mat bgr;
  Pose pose;
  GroundTruth evaluation;
};

struct FlowVector {
  cv::Point2f origin;
  cv::Point2f displacement;
  float confidence{};
  float ttc_proxy_s{std::numeric_limits<float>::infinity()};
};

struct PerceptionResult {
  std::vector<FlowVector> vectors;
  cv::Point2f focus_of_expansion;
  float expansion{};
  double cpu_ms{};
  double gpu_ms{};
  double total_ms{};
};

struct RiskAssessment {
  std::array<float, 3> sectors{};  // left, centre, right in [0, 1]
  float minimum_ttc_s{std::numeric_limits<float>::infinity()};
  WarningLevel warning{WarningLevel::Clear};
  std::vector<cv::Rect> clusters;
};

struct ControlCommand {
  float speed_mps{2.0F};
  float yaw_rate_deg_s{};
  float brake{};
};

struct StageLatencies {
  double camera_ms{};
  double decode_ms{};
  double perception_ms{};
  double risk_control_ms{};
  double actuation_ms{};
  double thermal_penalty_ms{};
  double render_ms{};
  double total_ms{};
};

struct VirtualHardwareState {
  std::string profile{"desktop-native"};
  bool simulated{};
  int frame_width{640};
  int frame_height{360};
  double target_fps{30.0};
  int declared_cpu_cores{};
  int declared_memory_mb{};
  double camera_latency_ms{};
  double actuation_latency_ms{};
  double thermal_penalty_ms{};
  double temperature_c{};
  bool throttled{};
  bool gpu_available{true};
  bool fallback_active{};
  bool frame_reused{};
  double deadline_ms{33.333};
  ControlCommand applied_command;
};

struct Telemetry {
  static constexpr int schema_version = 2;
  std::uint64_t frame_id{};
  double simulation_time_s{};
  std::string mode;
  Pose pose;
  StageLatencies latency;
  double fps{};
  bool deadline_missed{};
  float gpu_ratio{};
  RiskAssessment risk;
  ControlCommand command;
  VirtualHardwareState hardware;
  GroundTruth evaluation;
};

struct Thresholds {
  float yellow_ttc_s{3.0F};
  float red_ttc_s{1.5F};
};

std::string to_string(ExecutionMode mode);
ExecutionMode parse_mode(const std::string& value);
std::string to_string(WarningLevel warning);

}  // namespace flowguard
