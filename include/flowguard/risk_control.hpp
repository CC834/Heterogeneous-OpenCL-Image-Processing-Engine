#pragma once

#include "flowguard/types.hpp"

namespace flowguard {

class RiskEstimator {
 public:
  explicit RiskEstimator(Thresholds thresholds = {});
  RiskAssessment assess(const PerceptionResult& perception, cv::Size frame_size) const;

 private:
  Thresholds thresholds_;
};

class AvoidanceController {
 public:
  ControlCommand update(const RiskAssessment& risk, double dt_s);
  void reset();

 private:
  ControlCommand previous_;
  int turn_direction_{};
  int clear_frames_{};
  int centre_streak_{};
};

}  // namespace flowguard
