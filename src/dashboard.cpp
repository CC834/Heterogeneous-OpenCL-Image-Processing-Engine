#include "flowguard/dashboard.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace flowguard {
namespace {

std::string fixed(double value, int precision = 1) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

cv::Scalar warning_color(WarningLevel warning) {
  if (warning == WarningLevel::Red) return {40, 40, 245};
  if (warning == WarningLevel::Yellow) return {20, 210, 245};
  return {70, 210, 80};
}

}  // namespace

NativeDashboard::NativeDashboard(bool visible) : visible_(visible) {}

cv::Mat NativeDashboard::render(const cv::Mat& frame,
                                const PerceptionResult& perception,
                                const Telemetry& telemetry) {
  cv::Mat output = frame.clone();
  const cv::Scalar color = warning_color(telemetry.risk.warning);

  for (const auto& flow : perception.vectors) {
    if (flow.confidence < 0.12F) continue;
    cv::arrowedLine(output, flow.origin, flow.origin + flow.displacement, {255, 210, 40}, 1,
                    cv::LINE_AA, 0, 0.25);
  }
  if (perception.focus_of_expansion.x >= 0.0F) {
    cv::drawMarker(output, perception.focus_of_expansion, {255, 80, 255}, cv::MARKER_CROSS, 20, 2);
  }
  for (const auto& cluster : telemetry.risk.clusters) cv::rectangle(output, cluster, color, 2);

  const int sector_width = output.cols / 3;
  for (int i = 0; i < 3; ++i) {
    const int height = static_cast<int>(telemetry.risk.sectors[i] * 70.0F);
    cv::rectangle(output, {i * sector_width + 2, output.rows - height - 2,
                           sector_width - 4, height},
                  warning_color(telemetry.risk.sectors[i] > 0.55F ? WarningLevel::Red
                                : telemetry.risk.sectors[i] > 0.15F ? WarningLevel::Yellow
                                                                    : WarningLevel::Clear),
                  cv::FILLED);
  }

  cv::rectangle(output, {0, 0, output.cols, 58}, {18, 22, 28}, cv::FILLED);
  const std::string line1 = "FLOWGUARD  " + telemetry.mode + "  " +
                            fixed(telemetry.fps) + " FPS  " +
                            fixed(telemetry.latency.total_ms) + " ms";
  const std::string line2 = "TTC " +
      (std::isfinite(telemetry.risk.minimum_ttc_s) ? fixed(telemetry.risk.minimum_ttc_s, 2) + "s" : "--") +
      "   CPU/GPU " + fixed((1.0F - telemetry.gpu_ratio) * 100.0F, 0) + "/" +
      fixed(telemetry.gpu_ratio * 100.0F, 0) + "%   speed " +
      fixed(telemetry.command.speed_mps) + " yaw " + fixed(telemetry.command.yaw_rate_deg_s, 0);
  cv::putText(output, line1, {12, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv::LINE_AA);
  cv::putText(output, line2, {12, 47}, cv::FONT_HERSHEY_SIMPLEX, 0.43, {225, 225, 225}, 1,
              cv::LINE_AA);
  cv::circle(output, {output.cols - 22, 22}, 9, color, cv::FILLED);

  route_.push_back({static_cast<float>(telemetry.pose.x), static_cast<float>(telemetry.pose.y)});
  if (route_.size() > 120) route_.pop_front();
  const cv::Rect map(output.cols - 120, 64, 110, 74);
  cv::rectangle(output, map, {20, 25, 31}, cv::FILLED);
  for (std::size_t i = 1; i < route_.size(); ++i) {
    const auto map_point = [&](const cv::Point2f& p) {
      return cv::Point(map.x + 55 + static_cast<int>(p.y * 5.0F),
                       map.y + 66 - static_cast<int>(p.x * 4.0F));
    };
    cv::line(output, map_point(route_[i - 1]), map_point(route_[i]), {80, 220, 255}, 1);
  }
  return output;
}

bool NativeDashboard::show(const cv::Mat& dashboard) {
  if (!visible_) return true;
  cv::imshow("FlowGuard OpenCL", dashboard);
  const int key = cv::waitKey(1);
  return key != 27 && key != 'q';
}

}  // namespace flowguard
