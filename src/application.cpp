#include "flowguard/application.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "flowguard/artifacts.hpp"
#include "flowguard/dashboard.hpp"
#include "flowguard/opencl_pipeline.hpp"
#include "flowguard/protocol.hpp"
#include "flowguard/risk_control.hpp"
#include "flowguard/scheduler.hpp"
#include "flowguard/web_server.hpp"

namespace flowguard {
namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  ExecutionMode mode{ExecutionMode::Adaptive};
  float gpu_ratio{0.5F};
  Thresholds thresholds;
  int frames{180};
  int repeats{5};
  std::uint32_t seed{42};
  std::string scenario{"corridor"};
  std::string input;
  std::string synthetic;
  bool native_dashboard{};
  bool web_dashboard{};
  bool record{};
  bool visible_blender{};
  bool connect_only{};
};

std::string usage() {
  return R"(FlowGuard OpenCL 1.0.0

Usage:
  flowguard devices
  flowguard simulate --scenario corridor --mode adaptive --dashboard native,web --record
  flowguard replay --input flight.mp4 --mode fixed --gpu-ratio 0.70
  flowguard replay --synthetic expanding --frames 120 --mode cpu --no-dashboard
  flowguard benchmark --suite default --modes cpu,gpu,fixed,adaptive --repeats 5
  flowguard report --run artifacts/<run-id>

Options:
  --yellow-ttc SECONDS   Yellow warning threshold (default 3.0)
  --red-ttc SECONDS      Red warning threshold (default 1.5)
  --gpu-ratio RATIO      Fixed/adaptive initial GPU tile ratio (default 0.5)
  --seed INTEGER         Deterministic simulation seed (default 42)
  --visible              Run Blender with a visible window
  --connect-only         Wait for an already-running simulator
)";
}

bool has(const std::vector<std::string>& args, const std::string& flag) {
  return std::find(args.begin(), args.end(), flag) != args.end();
}

std::optional<std::string> value(const std::vector<std::string>& args, const std::string& flag) {
  const auto it = std::find(args.begin(), args.end(), flag);
  if (it == args.end()) return std::nullopt;
  if (std::next(it) == args.end() || std::next(it)->rfind("--", 0) == 0) {
    throw std::invalid_argument(flag + " requires a value");
  }
  return *std::next(it);
}

Options parse_options(const std::vector<std::string>& args) {
  Options options;
  if (auto v = value(args, "--mode")) options.mode = parse_mode(*v);
  if (auto v = value(args, "--gpu-ratio")) options.gpu_ratio = std::stof(*v);
  if (auto v = value(args, "--yellow-ttc")) options.thresholds.yellow_ttc_s = std::stof(*v);
  if (auto v = value(args, "--red-ttc")) options.thresholds.red_ttc_s = std::stof(*v);
  if (auto v = value(args, "--frames")) options.frames = std::stoi(*v);
  if (auto v = value(args, "--repeats")) options.repeats = std::stoi(*v);
  if (auto v = value(args, "--seed")) options.seed = static_cast<std::uint32_t>(std::stoul(*v));
  if (auto v = value(args, "--scenario")) options.scenario = *v;
  if (auto v = value(args, "--input")) options.input = *v;
  if (auto v = value(args, "--synthetic")) options.synthetic = *v;
  if (auto v = value(args, "--dashboard")) {
    options.native_dashboard = v->find("native") != std::string::npos;
    options.web_dashboard = v->find("web") != std::string::npos;
  }
  options.record = has(args, "--record");
  options.visible_blender = has(args, "--visible");
  options.connect_only = has(args, "--connect-only");
  if (has(args, "--no-dashboard")) options.native_dashboard = options.web_dashboard = false;
  if (options.gpu_ratio < 0.0F || options.gpu_ratio > 1.0F || options.frames < 2 ||
      options.repeats < 1) {
    throw std::invalid_argument("ratio, frame count, or repeat count is outside its valid range");
  }
  return options;
}

cv::Mat synthetic_frame(int index, const std::string& kind, cv::Size size = {640, 360}) {
  cv::Mat image(size, CV_8UC3, cv::Scalar(22, 28, 36));
  for (int y = 0; y < size.height; y += 24) cv::line(image, {0, y}, {size.width, y}, {34, 40, 48});
  for (int x = 0; x < size.width; x += 24) cv::line(image, {x, 0}, {x, size.height}, {34, 40, 48});
  const int shift = kind == "translation" ? index * 2 : 0;
  const float scale = kind == "expanding" ? 1.0F + index * 0.018F : 1.0F;
  const cv::Point centre(size.width / 2 + shift, size.height / 2);
  const int radius = static_cast<int>(45 * scale);
  cv::circle(image, centre, radius, {30, 90, 235}, cv::FILLED, cv::LINE_AA);
  cv::circle(image, centre, std::max(4, radius / 3), {230, 220, 60}, 3, cv::LINE_AA);
  cv::putText(image, "FLOWGUARD SYNTHETIC", {16, size.height - 18}, cv::FONT_HERSHEY_SIMPLEX,
              0.55, {190, 200, 215}, 1, cv::LINE_AA);
  return image;
}

std::string base64(const std::vector<unsigned char>& bytes) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const unsigned value = static_cast<unsigned>(bytes[i]) << 16 |
                           (i + 1 < bytes.size() ? static_cast<unsigned>(bytes[i + 1]) << 8 : 0) |
                           (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    output.push_back(alphabet[(value >> 18) & 63]);
    output.push_back(alphabet[(value >> 12) & 63]);
    output.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=');
    output.push_back(i + 2 < bytes.size() ? alphabet[value & 63] : '=');
  }
  return output;
}

struct RunSummary {
  std::vector<double> latencies;
  int deadline_misses{};
  int collisions{};
  double minimum_clearance{std::numeric_limits<double>::infinity()};
  float final_gpu_ratio{};
};

Telemetry process_pair(const SimulationFrame& frame, const cv::Mat& previous,
                       OpenClPerception& perception, TileScheduler& scheduler,
                       RiskEstimator& risk_estimator, AvoidanceController& controller,
                       PerceptionResult& perception_result) {
  const auto started = Clock::now();
  perception_result = perception.process(previous, frame.bgr, scheduler.gpu_ratio());
  const auto domain_started = Clock::now();
  RiskAssessment risk = risk_estimator.assess(perception_result, frame.bgr.size());
  ControlCommand command = controller.update(risk, 1.0 / 30.0);
  scheduler.observe(static_cast<std::size_t>((1.0F - scheduler.gpu_ratio()) * perception_result.vectors.size()),
                    perception_result.cpu_ms,
                    static_cast<std::size_t>(scheduler.gpu_ratio() * perception_result.vectors.size()),
                    perception_result.gpu_ms);
  Telemetry telemetry;
  telemetry.frame_id = frame.frame_id;
  telemetry.simulation_time_s = frame.simulation_time_s;
  telemetry.mode = "";
  telemetry.pose = frame.pose;
  telemetry.risk = std::move(risk);
  telemetry.command = command;
  telemetry.evaluation = frame.evaluation;
  telemetry.gpu_ratio = scheduler.gpu_ratio();
  telemetry.latency.perception_ms = perception_result.total_ms;
  telemetry.latency.risk_control_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - domain_started).count();
  telemetry.latency.total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  telemetry.fps = telemetry.latency.total_ms > 0.0 ? 1000.0 / telemetry.latency.total_ms : 0.0;
  telemetry.deadline_missed = telemetry.latency.total_ms > 33.333;
  return telemetry;
}

int run_frames(const Options& options, const std::vector<SimulationFrame>& frames,
               bool formal_benchmark, RunSummary* summary = nullptr) {
  if (frames.size() < 2) throw std::invalid_argument("at least two frames are required");
  OpenClPerception perception(options.mode);
  TileScheduler scheduler(options.mode, options.gpu_ratio);
  RiskEstimator risk(options.thresholds);
  AvoidanceController controller;
  NativeDashboard dashboard(options.native_dashboard && !formal_benchmark);
  std::unique_ptr<WebServer> web;
  if (options.web_dashboard && !formal_benchmark) {
    web = std::make_unique<WebServer>();
    web->start();
    std::cout << "Web dashboard: http://127.0.0.1:8080\n";
  }
  std::unique_ptr<RunArtifacts> artifacts;
  if (options.record && !formal_benchmark) {
    artifacts = std::make_unique<RunArtifacts>("artifacts", "replay", frames[0].bgr.size(), 30.0);
  }

  for (std::size_t i = 1; i < frames.size(); ++i) {
    PerceptionResult perception_result;
    Telemetry telemetry = process_pair(frames[i], frames[i - 1].bgr, perception, scheduler,
                                       risk, controller, perception_result);
    telemetry.mode = to_string(options.mode);
    cv::Mat annotated;
    if (!formal_benchmark && (options.native_dashboard || options.web_dashboard || artifacts)) {
      const auto render_started = Clock::now();
      annotated = dashboard.render(frames[i].bgr, perception_result, telemetry);
      telemetry.latency.render_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - render_started).count();
    }
    if (web) {
      std::string preview;
      if (i % 3 == 0) {
        std::vector<unsigned char> jpeg;
        cv::imencode(".jpg", annotated, jpeg, {cv::IMWRITE_JPEG_QUALITY, 72});
        preview = base64(jpeg);
      }
      web->publish(telemetry_json(telemetry), preview);
    }
    if (artifacts) artifacts->append(telemetry, annotated);
    if (!dashboard.show(annotated)) break;
    if (summary) {
      summary->latencies.push_back(telemetry.latency.total_ms);
      summary->deadline_misses += telemetry.deadline_missed;
      summary->collisions += telemetry.evaluation.collision;
      if (std::isfinite(telemetry.evaluation.nearest_obstacle_m)) {
        summary->minimum_clearance = std::min(summary->minimum_clearance,
                                             telemetry.evaluation.nearest_obstacle_m);
      }
      summary->final_gpu_ratio = telemetry.gpu_ratio;
    }
  }
  if (artifacts) std::cout << "Artifacts: " << artifacts->directory() << '\n';
  return 0;
}

std::vector<SimulationFrame> load_frames(const Options& options) {
  std::vector<SimulationFrame> frames;
  if (!options.synthetic.empty()) {
    for (int i = 0; i < options.frames; ++i) {
      frames.push_back({static_cast<std::uint64_t>(i), i / 30.0, options.seed,
                        synthetic_frame(i, options.synthetic), {}, {}});
    }
    return frames;
  }
  if (options.input.empty()) throw std::invalid_argument("replay requires --input or --synthetic");
  cv::VideoCapture video(options.input);
  if (!video.isOpened()) throw std::runtime_error("cannot open replay input: " + options.input);
  cv::Mat frame;
  std::uint64_t id = 0;
  while (id < static_cast<std::uint64_t>(options.frames) && video.read(frame)) {
    cv::resize(frame, frame, {640, 360});
    frames.push_back({id, id / 30.0, options.seed, frame.clone(), {}, {}});
    ++id;
  }
  return frames;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(fraction * (values.size() - 1))];
}

int benchmark(const std::vector<std::string>& args) {
  Options base = parse_options(args);
  base.synthetic = "expanding";
  base.frames = value(args, "--frames") ? base.frames : 62;
  const std::string modes = value(args, "--modes").value_or("cpu,gpu,fixed,adaptive");
  const auto frames = load_frames(base);  // one immutable input shared by every mode
  const std::string run_id = make_run_id();
  const std::filesystem::path directory = std::filesystem::path("artifacts") / run_id;
  std::filesystem::create_directories(directory);
  std::ofstream csv(directory / "benchmark.csv");
  csv << "mode,repeat,frames,p50_ms,p95_ms,p99_ms,deadline_miss_rate,throughput_fps,"
         "final_gpu_ratio,collision_rate,minimum_clearance_m,unnecessary_braking_rate\n";
  for (const std::string candidate : {"cpu", "gpu", "fixed", "adaptive"}) {
    if (modes.find(candidate) == std::string::npos) continue;
    for (int repeat = 1; repeat <= base.repeats; ++repeat) {
      Options options = base;
      options.mode = parse_mode(candidate);
      RunSummary result;
      run_frames(options, frames, true, &result);
      if (!result.latencies.empty()) result.latencies.erase(result.latencies.begin());  // warm-up
      const double total = std::accumulate(result.latencies.begin(), result.latencies.end(), 0.0);
      const double misses = static_cast<double>(std::count_if(result.latencies.begin(), result.latencies.end(),
                                                               [](double ms) { return ms > 33.333; }));
      csv << candidate << ',' << repeat << ',' << result.latencies.size() << ','
          << percentile(result.latencies, 0.50) << ',' << percentile(result.latencies, 0.95) << ','
          << percentile(result.latencies, 0.99) << ','
          << (result.latencies.empty() ? 0.0 : misses / result.latencies.size()) << ','
          << (total > 0.0 ? result.latencies.size() * 1000.0 / total : 0.0) << ','
          << result.final_gpu_ratio << ','
          << (result.latencies.empty() ? 0.0 : static_cast<double>(result.collisions) / result.latencies.size())
          << ',' << (std::isfinite(result.minimum_clearance) ? std::to_string(result.minimum_clearance) : "n/a")
          << ",n/a\n";
      std::cout << candidate << " repeat " << repeat << ": p95 "
                << percentile(result.latencies, 0.95) << " ms\n";
    }
  }
  std::ofstream metadata(directory / "metadata.json");
  metadata << "{\"schema_version\":1,\"workload\":\"deterministic synthetic expanding 640x360\","
              "\"warmup_frames\":1,\"deadline_ms\":33.333,\"note\":"
              "\"Modes ran sequentially on this host; this is not edge-device evidence.\",\"devices\":[";
  const auto devices = discover_opencl_devices();
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i) metadata << ',';
    metadata << "{\"type\":\"" << devices[i].type << "\",\"name\":\"" << devices[i].name
             << "\",\"platform\":\"" << devices[i].platform << "\",\"driver\":\""
             << devices[i].driver << "\"}";
  }
  metadata << "]}\n";
  std::cout << "Benchmark artifacts: " << directory << '\n';
  return 0;
}

int simulate(const std::vector<std::string>& args) {
  Options options = parse_options(args);
  const std::vector<std::string> scenarios = {"frontal-wall", "offset-pillar", "doorway",
                                               "corridor", "safe-lateral-pass", "crossing-obstacle"};
  if (std::find(scenarios.begin(), scenarios.end(), options.scenario) == scenarios.end()) {
    throw std::invalid_argument("unknown scenario: " + options.scenario);
  }
  SimulatorServer server;
  std::thread blender;
  if (!options.connect_only) {
    const std::string background = options.visible_blender ? "" : "--background ";
    const std::string command = "blender " + background + "--python \"" +
        std::string(FLOWGUARD_BLENDER_SCRIPT) + "\" -- --scenario " + options.scenario +
        " --seed " + std::to_string(options.seed) +
        " --frames " + std::to_string(options.frames);
    blender = std::thread([command] { std::system(command.c_str()); });
  }
  std::cout << "Waiting for Blender on 127.0.0.1:8765...\n";
  server.wait_for_client();

  OpenClPerception perception(options.mode);
  TileScheduler scheduler(options.mode, options.gpu_ratio);
  RiskEstimator risk(options.thresholds);
  AvoidanceController controller;
  NativeDashboard dashboard(options.native_dashboard);
  std::unique_ptr<WebServer> web;
  if (options.web_dashboard) { web = std::make_unique<WebServer>(); web->start(); }
  std::unique_ptr<RunArtifacts> artifacts;
  std::optional<SimulationFrame> previous;
  while (auto frame = server.receive()) {
    if (previous && frame->frame_id <= previous->frame_id) {
      throw std::runtime_error("stale or restarted frame ID received; reconnect to start a new run");
    }
    if (!previous) {
      previous = std::move(frame);
      server.send(previous->frame_id, {});
      if (options.record) artifacts = std::make_unique<RunArtifacts>("artifacts", "simulate " + options.scenario,
                                                                     previous->bgr.size(), 30.0);
      continue;
    }
    if (frame->bgr.size() != previous->bgr.size()) throw std::runtime_error("simulator dimensions changed mid-run");
    PerceptionResult result;
    Telemetry telemetry = process_pair(*frame, previous->bgr, perception, scheduler, risk, controller, result);
    telemetry.mode = to_string(options.mode);
    cv::Mat annotated = dashboard.render(frame->bgr, result, telemetry);
    server.send(frame->frame_id, telemetry.command);
    if (web) web->publish(telemetry_json(telemetry));
    if (artifacts) artifacts->append(telemetry, annotated);
    if (!dashboard.show(annotated)) break;
    previous = std::move(frame);
  }
  if (artifacts) std::cout << "Artifacts: " << artifacts->directory() << '\n';
  if (blender.joinable()) blender.join();
  return 0;
}

int generate_report(const std::vector<std::string>& args) {
  const auto run = value(args, "--run");
  if (!run) throw std::invalid_argument("report requires --run artifacts/<run-id>");
  const std::filesystem::path input(*run);
  if (!std::filesystem::exists(input / "telemetry.jsonl")) {
    throw std::runtime_error("run does not contain telemetry.jsonl: " + input.string());
  }
  const std::filesystem::path report = input / "report";
  std::filesystem::create_directories(report);
  const auto web_root = std::filesystem::path(FLOWGUARD_WEB_ROOT);
  if (!std::filesystem::exists(web_root / "index.html")) {
    throw std::runtime_error("web assets are not built; run npm run build in web/");
  }
  std::filesystem::copy(web_root, report,
                        std::filesystem::copy_options::recursive |
                            std::filesystem::copy_options::overwrite_existing);
  std::ifstream telemetry(input / "telemetry.jsonl");
  std::ofstream data(report / "data.js");
  data << "window.FLOWGUARD_REPLAY=[\n";
  std::string line;
  while (std::getline(telemetry, line)) data << line << ",\n";
  data << "];\n";
  if (std::filesystem::exists(input / "annotated.mp4")) {
    std::filesystem::copy_file(input / "annotated.mp4", report / "annotated.mp4",
                               std::filesystem::copy_options::overwrite_existing);
  }
  std::cout << "Interactive report: " << report / "index.html" << '\n';
  return 0;
}

}  // namespace

int run_application(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
    std::cout << usage();
    return argc < 2 ? 1 : 0;
  }
  const std::string command = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);
  if (command == "devices") {
    const auto devices = discover_opencl_devices();
    std::cout << "TYPE  DEVICE | PLATFORM | DRIVER\n";
    for (const auto& device : devices) {
      std::cout << std::left << std::setw(5) << device.type << " " << device.name << " | "
                << device.platform << " | " << device.driver << '\n';
    }
    return devices.empty() ? 1 : 0;
  }
  if (command == "replay") {
    Options options = parse_options(args);
    return run_frames(options, load_frames(options), false);
  }
  if (command == "benchmark") return benchmark(args);
  if (command == "simulate") return simulate(args);
  if (command == "report") return generate_report(args);
  throw std::invalid_argument("unknown command '" + command + "'\n" + usage());
}

}  // namespace flowguard
