#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "flowguard/types.hpp"

namespace flowguard {

class MessageFramer {
 public:
  explicit MessageFramer(std::size_t maximum_payload = 16 * 1024 * 1024);
  std::vector<std::vector<std::uint8_t>> feed(const std::uint8_t* data, std::size_t size);
  void reset();

 private:
  std::size_t maximum_payload_;
  std::vector<std::uint8_t> buffer_;
};

std::vector<std::uint8_t> frame_payload(const std::vector<std::uint8_t>& payload);
std::vector<std::uint8_t> encode_simulation_frame(const SimulationFrame& frame);
SimulationFrame decode_simulation_frame(const std::vector<std::uint8_t>& payload);
std::vector<std::uint8_t> encode_command(std::uint64_t frame_id,
                                         const ControlCommand& command);

class SimulatorServer {
 public:
  explicit SimulatorServer(std::uint16_t port = 8765);
  ~SimulatorServer();
  SimulatorServer(const SimulatorServer&) = delete;
  SimulatorServer& operator=(const SimulatorServer&) = delete;

  void wait_for_client();
  std::optional<SimulationFrame> receive();
  void send(std::uint64_t frame_id, const ControlCommand& command);

 private:
  int listen_fd_{-1};
  int client_fd_{-1};
  MessageFramer framer_;
  std::vector<std::vector<std::uint8_t>> pending_;
};

}  // namespace flowguard
