#include "flowguard/risk_control.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace flowguard {

RiskEstimator::RiskEstimator(Thresholds thresholds) : thresholds_(thresholds) {
  if (thresholds.yellow_ttc_s <= thresholds.red_ttc_s || thresholds.red_ttc_s <= 0.0F) {
    throw std::invalid_argument("TTC thresholds require yellow > red > 0");
  }
}

RiskAssessment RiskEstimator::assess(const PerceptionResult& perception,
                                     cv::Size frame_size) const {
  RiskAssessment result;
  cv::Mat danger = cv::Mat::zeros(frame_size, CV_8UC1);
  std::array<float, 3> weights{};

  for (const auto& vector : perception.vectors) {
    if (vector.confidence < 0.20F || !std::isfinite(vector.ttc_proxy_s) ||
        cv::norm(vector.displacement) < 0.75F ||
        vector.origin.y < frame_size.height * 0.10F ||
        vector.origin.y > frame_size.height * 0.80F) continue;
    const int sector = std::clamp(static_cast<int>(3.0F * vector.origin.x /
                                                   static_cast<float>(frame_size.width)),
                                  0, 2);
    const float severity = std::clamp((thresholds_.yellow_ttc_s - vector.ttc_proxy_s) /
                                          thresholds_.yellow_ttc_s,
                                      0.0F, 1.0F) * vector.confidence;
    result.sectors[sector] += severity;
    weights[sector] += vector.confidence;
    result.minimum_ttc_s = std::min(result.minimum_ttc_s, vector.ttc_proxy_s);
    if (vector.ttc_proxy_s < thresholds_.yellow_ttc_s) {
      cv::circle(danger, vector.origin, 11, cv::Scalar(255), -1);
    }
  }

  for (int i = 0; i < 3; ++i) {
    if (weights[i] > 0.0F) result.sectors[i] = std::clamp(result.sectors[i] / weights[i], 0.0F, 1.0F);
  }
  if (result.minimum_ttc_s < thresholds_.red_ttc_s) result.warning = WarningLevel::Red;
  else if (result.minimum_ttc_s < thresholds_.yellow_ttc_s) result.warning = WarningLevel::Yellow;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(danger, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (const auto& contour : contours) {
    const cv::Rect box = cv::boundingRect(contour);
    if (box.area() >= 100) result.clusters.push_back(box);
  }
  return result;
}

ControlCommand AvoidanceController::update(const RiskAssessment& risk, double dt_s) {
  const auto update_streak = [](int streak, bool active) {
    return active ? std::min(streak + 1, 30) : std::max(streak - 1, 0);
  };
  centre_streak_ = update_streak(centre_streak_, risk.sectors[1] > 0.18F);
  const bool centre_blocked = centre_streak_ >= 3;
  int desired_turn = 0;
  float desired_speed = 2.0F;
  float desired_brake = 0.0F;

  if (centre_blocked) {
    clear_frames_ = -7;  // preserve a confirmed turn for ten frames in total
    const float left = risk.sectors[0];
    const float right = risk.sectors[2];
    if (left > 0.65F && right > 0.65F) {
      desired_speed = 0.0F;
      desired_brake = 1.0F;
    } else {
      // Positive yaw turns toward image-left. Break exact ties toward image-right
      // so the seeded offset-pillar scenario does not steer into its +Y offset.
      desired_turn = left < right ? 1 : -1;
      desired_speed = risk.warning == WarningLevel::Red ? 0.65F : 1.35F;
    }
  } else if (++clear_frames_ < 5) {
    desired_turn = turn_direction_;
    if (turn_direction_ != 0) desired_speed = 1.0F;
  }

  if (desired_turn != 0) turn_direction_ = desired_turn;
  else if (clear_frames_ >= 5) turn_direction_ = 0;

  const float turn_rate = centre_blocked && risk.warning == WarningLevel::Red ? 60.0F : 45.0F;
  const float desired_yaw = static_cast<float>(turn_direction_) * turn_rate;
  const float dt = std::clamp(static_cast<float>(dt_s), 0.001F, 0.1F);
  const float speed_step = 3.0F * dt;
  previous_.speed_mps += std::clamp(desired_speed - previous_.speed_mps, -speed_step, speed_step);
  previous_.yaw_rate_deg_s += std::clamp(desired_yaw - previous_.yaw_rate_deg_s, -180.0F * dt, 180.0F * dt);
  previous_.brake += std::clamp(desired_brake - previous_.brake, -4.0F * dt, 4.0F * dt);
  return previous_;
}

void AvoidanceController::reset() {
  previous_ = {};
  turn_direction_ = 0;
  clear_frames_ = 0;
  centre_streak_ = 0;
}

std::string to_string(ExecutionMode mode) {
  switch (mode) {
    case ExecutionMode::Cpu: return "cpu";
    case ExecutionMode::Gpu: return "gpu";
    case ExecutionMode::Fixed: return "fixed";
    case ExecutionMode::Adaptive: return "adaptive";
  }
  return "unknown";
}

ExecutionMode parse_mode(const std::string& value) {
  if (value == "cpu") return ExecutionMode::Cpu;
  if (value == "gpu") return ExecutionMode::Gpu;
  if (value == "fixed") return ExecutionMode::Fixed;
  if (value == "adaptive") return ExecutionMode::Adaptive;
  throw std::invalid_argument("unknown execution mode: " + value);
}

std::string to_string(WarningLevel warning) {
  switch (warning) {
    case WarningLevel::Clear: return "clear";
    case WarningLevel::Yellow: return "yellow";
    case WarningLevel::Red: return "red";
  }
  return "unknown";
}

}  // namespace flowguard
