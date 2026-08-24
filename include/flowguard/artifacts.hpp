#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

#include "flowguard/types.hpp"

namespace flowguard {

std::string telemetry_json(const Telemetry& telemetry);

class RunArtifacts {
 public:
  RunArtifacts(const std::filesystem::path& root, const std::string& command,
               cv::Size frame_size, double fps, const VirtualHardwareState& hardware);
  ~RunArtifacts();

  void append(const Telemetry& telemetry, const cv::Mat& annotated_frame);
  void write_benchmark(const std::string& csv);
  const std::filesystem::path& directory() const;

 private:
  std::filesystem::path directory_;
  std::ofstream telemetry_;
  cv::VideoWriter video_;
};

std::string make_run_id();

}  // namespace flowguard
