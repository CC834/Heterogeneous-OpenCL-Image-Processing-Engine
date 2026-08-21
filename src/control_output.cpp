#include "flowguard/control_output.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace flowguard {

UdpEndpoint parse_loopback_endpoint(const std::string& value) {
  const auto separator = value.rfind(':');
  if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
    throw std::invalid_argument("control endpoint must be HOST:PORT");
  }
  UdpEndpoint endpoint{value.substr(0, separator), 0};
  if (endpoint.host != "127.0.0.1" && endpoint.host != "localhost") {
    throw std::invalid_argument("control endpoint must use loopback (127.0.0.1 or localhost)");
  }
  std::size_t consumed = 0;
  const unsigned long port = std::stoul(value.substr(separator + 1), &consumed);
  if (consumed != value.size() - separator - 1 || port == 0 ||
      port > std::numeric_limits<std::uint16_t>::max()) {
    throw std::invalid_argument("control endpoint port is invalid");
  }
  endpoint.port = static_cast<std::uint16_t>(port);
  return endpoint;
}

std::string control_command_json(std::uint64_t frame_id, double simulation_time_s,
                                 const ControlCommand& command) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(5)
         << "{\"schema_version\":1,\"source\":\"flowguard-opencl\",\"frame_id\":"
         << frame_id << ",\"simulation_time_s\":" << simulation_time_s
         << ",\"speed_mps\":" << command.speed_mps
         << ",\"yaw_rate_deg_s\":" << command.yaw_rate_deg_s
         << ",\"brake\":" << command.brake << '}';
  return output.str();
}

UdpControlOutput::UdpControlOutput(const std::string& value) {
  const UdpEndpoint endpoint = parse_loopback_endpoint(value);
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_ < 0) throw std::runtime_error("failed to create UDP control socket: " +
                                            std::string(std::strerror(errno)));
  in_addr address{};
  if (::inet_pton(AF_INET, "127.0.0.1", &address) != 1) {
    ::close(socket_);
    socket_ = -1;
    throw std::runtime_error("failed to resolve loopback control endpoint");
  }
  address_ = address.s_addr;
  port_ = endpoint.port;
}

UdpControlOutput::~UdpControlOutput() {
  if (socket_ >= 0) ::close(socket_);
}

void UdpControlOutput::send(std::uint64_t frame_id, double simulation_time_s,
                            const ControlCommand& command) const {
  const std::string payload = control_command_json(frame_id, simulation_time_s, command);
  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_addr.s_addr = address_;
  destination.sin_port = htons(port_);
  const auto sent = ::sendto(socket_, payload.data(), payload.size(), 0,
                             reinterpret_cast<const sockaddr*>(&destination),
                             sizeof(destination));
  if (sent < 0 || static_cast<std::size_t>(sent) != payload.size()) {
    throw std::runtime_error("failed to send UDP control command: " +
                             std::string(std::strerror(errno)));
  }
}

}  // namespace flowguard
