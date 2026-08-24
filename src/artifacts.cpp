#include "flowguard/artifacts.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <CL/cl.h>

namespace flowguard {
namespace {

std::string json_float(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::fixed << std::setprecision(5) << value;
  return out.str();
}

}  // namespace

std::string telemetry_json(const Telemetry& t) {
  std::ostringstream out;
  out << "{\"schema_version\":" << Telemetry::schema_version
      << ",\"frame_id\":" << t.frame_id
      << ",\"simulation_time_s\":" << json_float(t.simulation_time_s)
      << ",\"mode\":\"" << t.mode << "\""
      << ",\"pose\":{\"x\":" << json_float(t.pose.x) << ",\"y\":" << json_float(t.pose.y)
      << ",\"z\":" << json_float(t.pose.z) << ",\"yaw\":" << json_float(t.pose.yaw) << "}"
      << ",\"latency_ms\":{\"camera\":" << json_float(t.latency.camera_ms)
      << ",\"decode\":" << json_float(t.latency.decode_ms)
      << ",\"perception\":" << json_float(t.latency.perception_ms)
      << ",\"risk_control\":" << json_float(t.latency.risk_control_ms)
      << ",\"actuation\":" << json_float(t.latency.actuation_ms)
      << ",\"thermal_penalty\":" << json_float(t.latency.thermal_penalty_ms)
      << ",\"render\":" << json_float(t.latency.render_ms)
      << ",\"total\":" << json_float(t.latency.total_ms) << "}"
      << ",\"fps\":" << json_float(t.fps)
      << ",\"deadline_missed\":" << (t.deadline_missed ? "true" : "false")
      << ",\"gpu_ratio\":" << json_float(t.gpu_ratio)
      << ",\"sector_risk\":[" << json_float(t.risk.sectors[0]) << ','
      << json_float(t.risk.sectors[1]) << ',' << json_float(t.risk.sectors[2]) << ']'
      << ",\"minimum_ttc_s\":" << json_float(t.risk.minimum_ttc_s)
      << ",\"warning\":\"" << to_string(t.risk.warning) << "\""
      << ",\"command\":{\"speed\":" << json_float(t.command.speed_mps)
      << ",\"yaw_rate\":" << json_float(t.command.yaw_rate_deg_s)
      << ",\"brake\":" << json_float(t.command.brake) << "}"
      << ",\"virtual_hardware\":{\"profile\":\"" << t.hardware.profile
      << "\",\"simulated\":" << (t.hardware.simulated ? "true" : "false")
      << ",\"camera\":{\"width\":" << t.hardware.frame_width
      << ",\"height\":" << t.hardware.frame_height
      << ",\"target_fps\":" << json_float(t.hardware.target_fps) << "}"
      << ",\"declared_cpu_cores\":" << t.hardware.declared_cpu_cores
      << ",\"declared_memory_mb\":" << t.hardware.declared_memory_mb
      << ",\"temperature_c\":" << json_float(t.hardware.temperature_c)
      << ",\"throttled\":" << (t.hardware.throttled ? "true" : "false")
      << ",\"gpu_available\":" << (t.hardware.gpu_available ? "true" : "false")
      << ",\"fallback_active\":" << (t.hardware.fallback_active ? "true" : "false")
      << ",\"frame_reused\":" << (t.hardware.frame_reused ? "true" : "false")
      << ",\"deadline_ms\":" << json_float(t.hardware.deadline_ms)
      << ",\"applied_command\":{\"speed\":"
      << json_float(t.hardware.applied_command.speed_mps)
      << ",\"yaw_rate\":" << json_float(t.hardware.applied_command.yaw_rate_deg_s)
      << ",\"brake\":" << json_float(t.hardware.applied_command.brake) << "}}"
      << ",\"evaluation_only\":{\"nearest_obstacle_m\":"
      << json_float(t.evaluation.nearest_obstacle_m) << ",\"true_ttc_s\":"
      << json_float(t.evaluation.time_to_contact_s) << ",\"collision\":"
      << (t.evaluation.collision ? "true" : "false") << "}}";
  return out.str();
}

std::string make_run_id() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::ostringstream out;
  out << std::put_time(std::localtime(&value), "%Y%m%d-%H%M%S");
  return out.str();
}

RunArtifacts::RunArtifacts(const std::filesystem::path& root, const std::string& command,
                           cv::Size frame_size, double fps,
                           const VirtualHardwareState& hardware)
    : directory_(root / make_run_id()) {
  std::filesystem::create_directories(directory_ / "report");
  telemetry_.open(directory_ / "telemetry.jsonl");
  if (!telemetry_) throw std::runtime_error("failed to create telemetry artifact");
  video_.open((directory_ / "annotated.mp4").string(),
              cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame_size);
  if (!video_.isOpened()) {
    video_.open((directory_ / "annotated.mp4").string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame_size);
  }
  std::ofstream metadata(directory_ / "metadata.json");
  metadata << "{\"schema_version\":1,\"command\":\"" << command
           << "\",\"frame_width\":" << frame_size.width
           << ",\"frame_height\":" << frame_size.height
           << ",\"hardware_profile\":\"" << hardware.profile << "\""
           << ",\"hardware_simulated\":" << (hardware.simulated ? "true" : "false")
           << ",\"evidence_note\":\"Host OpenCL timing; virtual profile delays are simulated\""
           << ",\"opencl_header_version\":\"" << CL_VERSION_MAJOR(CL_TARGET_OPENCL_VERSION)
           << '.' << CL_VERSION_MINOR(CL_TARGET_OPENCL_VERSION) << "\"}\n";
}

RunArtifacts::~RunArtifacts() = default;

void RunArtifacts::append(const Telemetry& telemetry, const cv::Mat& annotated_frame) {
  telemetry_ << telemetry_json(telemetry) << '\n';
  telemetry_.flush();
  if (video_.isOpened() && !annotated_frame.empty()) video_.write(annotated_frame);
}

void RunArtifacts::write_benchmark(const std::string& csv) {
  std::ofstream output(directory_ / "benchmark.csv");
  output << csv;
}

const std::filesystem::path& RunArtifacts::directory() const { return directory_; }

}  // namespace flowguard
