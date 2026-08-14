#include "test_support.hpp"

#include <cmath>

#include <opencv2/core.hpp>

#include "flowguard/hardware_profile.hpp"

using namespace flowguard;

TEST_CASE("hardware profile lookup rejects unknown names") {
  bool rejected = false;
  try {
    (void)find_hardware_profile("imaginary-board");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  flowguard::test::require(rejected, "unknown profile should be rejected");
}

TEST_CASE("constrained camera is deterministic and reuses frames at 20 fps") {
  const auto profile = find_hardware_profile("edge-constrained-sim");
  HardwareEmulator first(profile, 7);
  HardwareEmulator second(profile, 7);
  cv::Mat source(360, 640, CV_8UC3, cv::Scalar(80, 120, 160));
  VirtualHardwareState first_state;
  VirtualHardwareState second_state;
  const cv::Mat first_capture = first.capture(source, 0, 0.0, first_state);
  const cv::Mat second_capture = second.capture(source, 0, 0.0, second_state);
  flowguard::test::require(first_capture.size() == cv::Size(320, 180),
                           "profile resolution was not applied");
  flowguard::test::require(cv::norm(first_capture, second_capture, cv::NORM_INF) == 0.0,
                           "seeded sensor transform should be deterministic");
  const cv::Mat reused = first.capture(cv::Mat::zeros(source.size(), source.type()), 1,
                                       1.0 / 30.0, first_state);
  flowguard::test::require(first_state.frame_reused, "20 fps profile should reuse this frame");
  flowguard::test::require(cv::norm(first_capture, reused, cv::NORM_INF) == 0.0,
                           "reused frame should match the last camera sample");
  flowguard::test::require(!first_state.gpu_available, "CPU-only profile exposed a GPU");
}

TEST_CASE("thermal and GPU fault transitions use simulation time") {
  HardwareEmulator thermal(find_hardware_profile("thermal-throttle-sim"));
  HardwareEmulator failure(find_hardware_profile("gpu-failure-sim"));
  VirtualHardwareState state;
  cv::Mat source(12, 12, CV_8UC3, cv::Scalar(0, 0, 0));
  (void)thermal.capture(source, 61, 2.1, state);
  flowguard::test::require(state.throttled, "thermal profile did not enter throttled state");
  flowguard::test::require(failure.gpu_available(1.99), "GPU failed too early");
  flowguard::test::require(!failure.gpu_available(2.0), "GPU did not fail at declared time");
}

TEST_CASE("sub-frame actuator latency delivers the current command") {
  HardwareEmulator hardware(find_hardware_profile("edge-balanced-sim"));
  VirtualHardwareState state;
  const ControlCommand requested{1.25F, -15.0F, 0.2F};
  const ControlCommand applied = hardware.deliver(requested, 0.1, state);
  flowguard::test::require(std::abs(applied.speed_mps - requested.speed_mps) < 0.001F,
                           "sub-frame actuator delay added an unintended full frame");
  flowguard::test::require(state.actuation_latency_ms >= 3.0,
                           "declared actuator latency was not applied");
}
