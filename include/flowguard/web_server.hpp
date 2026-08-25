#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace flowguard {

class WebServer {
 public:
  explicit WebServer(std::uint16_t port = 8080);
  ~WebServer();
  WebServer(const WebServer&) = delete;
  WebServer& operator=(const WebServer&) = delete;

  void start();
  void stop();
  void publish(std::string telemetry_json, std::string jpeg_base64 = {});

 private:
  void accept_loop();
  void handle(int socket_fd);

  std::uint16_t port_;
  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  std::thread thread_;
  std::mutex latest_mutex_;
  std::string latest_{"{\"schema_version\":2,\"status\":\"waiting\"}"};
};

}  // namespace flowguard
