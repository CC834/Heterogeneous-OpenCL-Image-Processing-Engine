#pragma once

#include <deque>

#include "flowguard/types.hpp"

namespace flowguard {

class NativeDashboard {
 public:
  explicit NativeDashboard(bool visible);
  cv::Mat render(const cv::Mat& frame, const PerceptionResult& perception,
                 const Telemetry& telemetry);
  bool show(const cv::Mat& dashboard);

 private:
  bool visible_;
  std::deque<cv::Point2f> route_;
};

}  // namespace flowguard
