#pragma once

#include <memory>
#include <string>
#include <vector>

#include "flowguard/types.hpp"

namespace flowguard {

struct OpenClDeviceInfo {
  std::string type;
  std::string name;
  std::string vendor;
  std::string driver;
  std::string platform;
};

struct KernelValidation {
  double grayscale_max_error{};
  double gaussian_max_error{};
  double pyramid_level1_max_error{};
  double pyramid_level2_max_error{};
};

class OpenClPerception {
 public:
  explicit OpenClPerception(ExecutionMode mode);
  ~OpenClPerception();
  OpenClPerception(OpenClPerception&&) noexcept;
  OpenClPerception& operator=(OpenClPerception&&) noexcept;
  OpenClPerception(const OpenClPerception&) = delete;
  OpenClPerception& operator=(const OpenClPerception&) = delete;

  PerceptionResult process(const cv::Mat& previous_bgr, const cv::Mat& current_bgr,
                           float gpu_ratio);
  const std::vector<OpenClDeviceInfo>& devices() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<OpenClDeviceInfo> discover_opencl_devices();
KernelValidation validate_opencl_preprocessing();

}  // namespace flowguard
