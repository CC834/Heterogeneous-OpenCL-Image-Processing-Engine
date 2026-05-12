#include "flowguard/protocol.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace flowguard {
namespace {

std::uint32_t read_u32(const std::uint8_t* bytes) {
  std::uint32_t network{};
  std::memcpy(&network, bytes, sizeof(network));
  return ntohl(network);
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
  const std::uint32_t network = htonl(value);
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&network);
  output.insert(output.end(), bytes, bytes + sizeof(network));
}

double json_number(const std::string& json, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::runtime_error("simulator metadata is missing numeric field '" + key + "'");
  }
  return std::stod(match[1].str());
}

bool json_bool(const std::string& json, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::runtime_error("simulator metadata is missing boolean field '" + key + "'");
  }
  return match[1].str() == "true";
}

void write_all(int fd, const std::vector<std::uint8_t>& bytes) {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t count = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (count <= 0) throw std::runtime_error("simulator connection closed while sending");
    sent += static_cast<std::size_t>(count);
  }
}

}  // namespace

MessageFramer::MessageFramer(std::size_t maximum_payload)
    : maximum_payload_(maximum_payload) {}

std::vector<std::vector<std::uint8_t>> MessageFramer::feed(const std::uint8_t* data,
                                                            std::size_t size) {
  buffer_.insert(buffer_.end(), data, data + size);
  std::vector<std::vector<std::uint8_t>> messages;
  while (buffer_.size() >= 4) {
    const std::size_t payload_size = read_u32(buffer_.data());
    if (payload_size > maximum_payload_) {
      reset();
      throw std::runtime_error("simulator payload exceeds configured maximum");
    }
    if (buffer_.size() < payload_size + 4) break;
    messages.emplace_back(buffer_.begin() + 4, buffer_.begin() + 4 + payload_size);
    buffer_.erase(buffer_.begin(), buffer_.begin() + 4 + payload_size);
  }
  return messages;
}

void MessageFramer::reset() { buffer_.clear(); }

std::vector<std::uint8_t> frame_payload(const std::vector<std::uint8_t>& payload) {
  if (payload.size() > UINT32_MAX) throw std::length_error("payload is too large");
  std::vector<std::uint8_t> framed;
  framed.reserve(payload.size() + 4);
  append_u32(framed, static_cast<std::uint32_t>(payload.size()));
  framed.insert(framed.end(), payload.begin(), payload.end());
  return framed;
}

std::vector<std::uint8_t> encode_simulation_frame(const SimulationFrame& frame) {
  if (frame.bgr.type() != CV_8UC3 || !frame.bgr.isContinuous()) {
    throw std::invalid_argument("simulation frame must be continuous 8-bit BGR");
  }
  std::ostringstream json;
  json << "{\"schema_version\":1,\"frame_id\":" << frame.frame_id
       << ",\"simulation_time_s\":" << frame.simulation_time_s
       << ",\"seed\":" << frame.seed << ",\"width\":" << frame.bgr.cols
       << ",\"height\":" << frame.bgr.rows << ",\"channels\":3"
       << ",\"x\":" << frame.pose.x << ",\"y\":" << frame.pose.y
       << ",\"z\":" << frame.pose.z << ",\"yaw\":" << frame.pose.yaw
       << ",\"nearest_obstacle_m\":" << frame.evaluation.nearest_obstacle_m
       << ",\"true_ttc_s\":" << frame.evaluation.time_to_contact_s
       << ",\"collision\":" << (frame.evaluation.collision ? "true" : "false") << "}";
  const std::string metadata = json.str();
  std::vector<std::uint8_t> payload;
  append_u32(payload, static_cast<std::uint32_t>(metadata.size()));
  payload.insert(payload.end(), metadata.begin(), metadata.end());
  payload.insert(payload.end(), frame.bgr.data,
                 frame.bgr.data + frame.bgr.total() * frame.bgr.elemSize());
  return frame_payload(payload);
}

SimulationFrame decode_simulation_frame(const std::vector<std::uint8_t>& payload) {
  if (payload.size() < 4) throw std::runtime_error("truncated simulator frame header");
  const std::size_t json_size = read_u32(payload.data());
  if (json_size > payload.size() - 4) throw std::runtime_error("truncated simulator metadata");
  const std::string json(payload.begin() + 4, payload.begin() + 4 + json_size);
  const int width = static_cast<int>(json_number(json, "width"));
  const int height = static_cast<int>(json_number(json, "height"));
  const int channels = static_cast<int>(json_number(json, "channels"));
  if (width <= 0 || height <= 0 || channels != 3 || width > 4096 || height > 4096) {
    throw std::runtime_error("invalid simulator frame dimensions");
  }
  const std::size_t image_size = static_cast<std::size_t>(width) * height * channels;
  if (payload.size() != 4 + json_size + image_size) {
    throw std::runtime_error("simulator image size does not match metadata");
  }

  SimulationFrame frame;
  frame.frame_id = static_cast<std::uint64_t>(json_number(json, "frame_id"));
  frame.simulation_time_s = json_number(json, "simulation_time_s");
  frame.seed = static_cast<std::uint32_t>(json_number(json, "seed"));
  frame.pose = {json_number(json, "x"), json_number(json, "y"),
                json_number(json, "z"), json_number(json, "yaw")};
  frame.evaluation = {json_number(json, "nearest_obstacle_m"),
                      json_number(json, "true_ttc_s"), json_bool(json, "collision")};
  cv::Mat view(height, width, CV_8UC3,
               const_cast<std::uint8_t*>(payload.data() + 4 + json_size));
  frame.bgr = view.clone();
  return frame;
}

std::vector<std::uint8_t> encode_command(std::uint64_t frame_id,
                                         const ControlCommand& command) {
  std::ostringstream json;
  json << "{\"schema_version\":1,\"frame_id\":" << frame_id
       << ",\"speed\":" << command.speed_mps
       << ",\"yaw_rate\":" << command.yaw_rate_deg_s
       << ",\"brake\":" << command.brake << "}";
  const std::string value = json.str();
  return frame_payload({value.begin(), value.end()});
}

SimulatorServer::SimulatorServer(std::uint16_t port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) throw std::runtime_error("failed to create simulator socket");
  int reuse = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(listen_fd_, 1) != 0) {
    ::close(listen_fd_);
    throw std::runtime_error("failed to bind simulator server to 127.0.0.1:" + std::to_string(port));
  }
}

SimulatorServer::~SimulatorServer() {
  if (client_fd_ >= 0) ::close(client_fd_);
  if (listen_fd_ >= 0) ::close(listen_fd_);
}

void SimulatorServer::wait_for_client() {
  client_fd_ = ::accept(listen_fd_, nullptr, nullptr);
  if (client_fd_ < 0) throw std::runtime_error("failed to accept Blender connection");
  framer_.reset();
  pending_.clear();
}

std::optional<SimulationFrame> SimulatorServer::receive() {
  while (pending_.empty()) {
    std::array<std::uint8_t, 65536> bytes{};
    const ssize_t count = ::recv(client_fd_, bytes.data(), bytes.size(), 0);
    if (count == 0) return std::nullopt;
    if (count < 0) throw std::runtime_error("failed to receive simulator frame");
    auto messages = framer_.feed(bytes.data(), static_cast<std::size_t>(count));
    pending_.insert(pending_.end(), std::make_move_iterator(messages.begin()),
                    std::make_move_iterator(messages.end()));
  }
  auto payload = std::move(pending_.front());
  pending_.erase(pending_.begin());
  return decode_simulation_frame(payload);
}

void SimulatorServer::send(std::uint64_t frame_id, const ControlCommand& command) {
  write_all(client_fd_, encode_command(frame_id, command));
}

}  // namespace flowguard
