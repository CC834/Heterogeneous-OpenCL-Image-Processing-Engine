#pragma once

#include <cstdint>
#include <string>

#include "flowguard/types.hpp"

namespace flowguard {

struct UdpEndpoint {
  std::string host;
  std::uint16_t port{};
};

UdpEndpoint parse_loopback_endpoint(const std::string& value);
std::string control_command_json(std::uint64_t frame_id, double simulation_time_s,
                                 const ControlCommand& command);

class UdpControlOutput {
 public:
  explicit UdpControlOutput(const std::string& endpoint);
  ~UdpControlOutput();
  UdpControlOutput(const UdpControlOutput&) = delete;
  UdpControlOutput& operator=(const UdpControlOutput&) = delete;

  void send(std::uint64_t frame_id, double simulation_time_s,
            const ControlCommand& command) const;

 private:
  int socket_{-1};
  std::uint32_t address_{};
  std::uint16_t port_{};
};

}  // namespace flowguard
