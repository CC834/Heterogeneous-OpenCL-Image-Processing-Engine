#include "test_support.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>

#include "flowguard/control_output.hpp"

using namespace flowguard;

TEST_CASE("control bridge accepts only loopback endpoints") {
  const UdpEndpoint endpoint = parse_loopback_endpoint("127.0.0.1:9002");
  flowguard::test::require(endpoint.port == 9002, "endpoint port was not parsed");
  bool rejected = false;
  try {
    (void)parse_loopback_endpoint("0.0.0.0:9002");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  flowguard::test::require(rejected, "non-loopback output should be rejected");
}

TEST_CASE("control bridge JSON carries a versioned command") {
  const std::string json = control_command_json(12, 0.4, {1.5F, -20.0F, 0.25F});
  flowguard::test::require(json.find("\"schema_version\":1") != std::string::npos,
                           "schema version missing");
  flowguard::test::require(json.find("\"frame_id\":12") != std::string::npos,
                           "frame ID missing");
  flowguard::test::require(json.find("\"yaw_rate_deg_s\":-20.00000") != std::string::npos,
                           "yaw command missing");
}

TEST_CASE("control output sends a loopback UDP datagram") {
  const int receiver = ::socket(AF_INET, SOCK_DGRAM, 0);
  flowguard::test::require(receiver >= 0, "failed to create test UDP receiver");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  flowguard::test::require(
      ::bind(receiver, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
      "failed to bind test UDP receiver");
  socklen_t address_size = sizeof(address);
  flowguard::test::require(
      ::getsockname(receiver, reinterpret_cast<sockaddr*>(&address), &address_size) == 0,
      "failed to inspect test UDP receiver");
  timeval timeout{1, 0};
  (void)::setsockopt(receiver, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  UdpControlOutput output("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
  output.send(9, 0.3, {1.25F, 15.0F, 0.1F});
  char payload[1024]{};
  const auto received = ::recv(receiver, payload, sizeof(payload), 0);
  ::close(receiver);
  flowguard::test::require(received > 0, "no loopback control datagram was received");
  const std::string json(payload, static_cast<std::size_t>(received));
  flowguard::test::require(json.find("\"frame_id\":9") != std::string::npos,
                           "received datagram had the wrong frame ID");
}
