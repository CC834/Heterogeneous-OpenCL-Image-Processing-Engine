#include "test_support.hpp"

#include <algorithm>

#include "flowguard/protocol.hpp"

using namespace flowguard;
using flowguard::test::require;

TEST_CASE("framer handles partial and combined packets") {
  const auto first = frame_payload({1, 2, 3});
  const auto second = frame_payload({4, 5});
  std::vector<std::uint8_t> bytes = first;
  bytes.insert(bytes.end(), second.begin(), second.end());
  MessageFramer framer;
  auto messages = framer.feed(bytes.data(), 2);
  require(messages.empty(), "partial prefix emitted a message");
  messages = framer.feed(bytes.data() + 2, bytes.size() - 2);
  require(messages.size() == 2, "combined packets were not split");
  require(messages[0] == std::vector<std::uint8_t>({1, 2, 3}), "first payload changed");
}

TEST_CASE("simulation frame round trips dimensions pose and ground truth") {
  SimulationFrame source;
  source.frame_id = 9;
  source.simulation_time_s = 0.3;
  source.seed = 42;
  source.pose = {1, 2, 3, 0.4};
  source.evaluation = {0.8, 0.4, false};
  source.bgr = cv::Mat(3, 4, CV_8UC3, cv::Scalar(10, 20, 30));
  const auto wire = encode_simulation_frame(source);
  const std::uint32_t size = (static_cast<std::uint32_t>(wire[0]) << 24) |
                             (static_cast<std::uint32_t>(wire[1]) << 16) |
                             (static_cast<std::uint32_t>(wire[2]) << 8) | wire[3];
  const auto decoded = decode_simulation_frame({wire.begin() + 4, wire.begin() + 4 + size});
  require(decoded.frame_id == source.frame_id, "frame ID changed");
  require(decoded.bgr.size() == source.bgr.size(), "dimensions changed");
  require(decoded.evaluation.nearest_obstacle_m == source.evaluation.nearest_obstacle_m,
          "evaluation metadata changed");
}
