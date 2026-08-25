#include "flowguard/web_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

namespace flowguard {
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

WebServer::WebServer(std::uint16_t port) : port_(port) {}
WebServer::~WebServer() { stop(); }

void WebServer::start() {
  if (running_.exchange(true)) return;
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  int reuse = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port_);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(listen_fd_, 8) != 0) {
    running_ = false;
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("failed to bind web dashboard to 127.0.0.1:" + std::to_string(port_));
  }
  thread_ = std::thread(&WebServer::accept_loop, this);
}

void WebServer::stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (thread_.joinable()) thread_.join();
}

void WebServer::publish(std::string telemetry_json, std::string jpeg_base64) {
  std::lock_guard<std::mutex> lock(latest_mutex_);
  if (!jpeg_base64.empty() && telemetry_json.size() > 1 && telemetry_json.back() == '}') {
    telemetry_json.pop_back();
    telemetry_json += ",\"preview_jpeg\":\"" + jpeg_base64 + "\"}";
  }
  latest_ = std::move(telemetry_json);
}

void WebServer::accept_loop() {
  while (running_) {
    const int socket_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (socket_fd < 0) continue;
    std::thread(&WebServer::handle, this, socket_fd).detach();
  }
}

void WebServer::handle(int socket_fd) {
  boost::asio::io_context context;
  tcp::socket socket(context, tcp::v4(), socket_fd);
  beast::flat_buffer buffer;
  http::request<http::string_body> request;
  beast::error_code error;
  http::read(socket, buffer, request, error);
  if (error) return;

  if (websocket::is_upgrade(request) && request.target() == "/ws") {
    websocket::stream<tcp::socket> ws(std::move(socket));
    ws.accept(request, error);
    while (running_ && !error) {
      std::string value;
      {
        std::lock_guard<std::mutex> lock(latest_mutex_);
        value = latest_;
      }
      ws.write(boost::asio::buffer(value), error);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return;
  }

  std::string target = request.target() == "/" ? "index.html" : std::string(request.target().substr(1));
  if (target.find("..") != std::string::npos) target.clear();
  const std::string body = target.empty() ? std::string{} : read_file(std::filesystem::path(FLOWGUARD_WEB_ROOT) / target);
  http::response<http::string_body> response(body.empty() ? http::status::not_found : http::status::ok,
                                             request.version());
  response.set(http::field::server, "FlowGuard OpenCL");
  response.set(http::field::content_type,
               target.size() >= 3 && target.substr(target.size() - 3) == ".js"
                   ? "text/javascript"
                   : target.size() >= 4 && target.substr(target.size() - 4) == ".css"
                         ? "text/css"
                         : "text/html");
  response.body() = body.empty() ? "FlowGuard dashboard assets are not built. Run npm run build in web/." : body;
  response.prepare_payload();
  http::write(socket, response, error);
}

}  // namespace flowguard
