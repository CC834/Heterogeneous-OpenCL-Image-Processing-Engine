#include "flowguard/opencl_pipeline.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace flowguard {
namespace {

using Clock = std::chrono::steady_clock;

void check(cl_int code, const std::string& operation) {
  if (code != CL_SUCCESS) {
    throw std::runtime_error(operation + " failed with OpenCL error " + std::to_string(code));
  }
}

std::string device_string(cl_device_id device, cl_device_info key) {
  std::size_t size{};
  check(clGetDeviceInfo(device, key, 0, nullptr, &size), "query OpenCL device string size");
  std::string value(size, '\0');
  check(clGetDeviceInfo(device, key, size, value.data(), nullptr), "query OpenCL device string");
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string platform_string(cl_platform_id platform, cl_platform_info key) {
  std::size_t size{};
  check(clGetPlatformInfo(platform, key, 0, nullptr, &size), "query OpenCL platform string size");
  std::string value(size, '\0');
  check(clGetPlatformInfo(platform, key, size, value.data(), nullptr), "query OpenCL platform string");
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

struct Candidate {
  cl_platform_id platform{};
  cl_device_id device{};
  cl_device_type type{};
};

std::vector<Candidate> candidates() {
  cl_uint count{};
  check(clGetPlatformIDs(0, nullptr, &count), "enumerate OpenCL platforms");
  std::vector<cl_platform_id> platforms(count);
  check(clGetPlatformIDs(count, platforms.data(), nullptr), "read OpenCL platforms");
  std::vector<Candidate> result;
  for (auto platform : platforms) {
    cl_uint device_count{};
    const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    check(status, "enumerate OpenCL devices");
    std::vector<cl_device_id> devices(device_count);
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr),
          "read OpenCL devices");
    for (auto device : devices) {
      cl_device_type type{};
      check(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr),
            "query OpenCL device type");
      result.push_back({platform, device, type});
    }
  }
  return result;
}

OpenClDeviceInfo info(const Candidate& candidate) {
  return {(candidate.type & CL_DEVICE_TYPE_GPU) ? "gpu" :
          (candidate.type & CL_DEVICE_TYPE_CPU) ? "cpu" : "other",
          device_string(candidate.device, CL_DEVICE_NAME),
          device_string(candidate.device, CL_DEVICE_VENDOR),
          device_string(candidate.device, CL_DRIVER_VERSION),
          platform_string(candidate.platform, CL_PLATFORM_NAME)};
}

std::string kernel_source() {
  std::ifstream input(FLOWGUARD_KERNEL_PATH);
  if (!input) throw std::runtime_error("cannot open OpenCL kernel at " FLOWGUARD_KERNEL_PATH);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct WorkerResult {
  std::vector<cl_float4> flow;
  int row_begin{};
  int row_end{};
  double elapsed_ms{};
};

class Worker {
 public:
  explicit Worker(const Candidate& candidate) : candidate_(candidate), info_(info(candidate)) {
    cl_int error{};
    context_ = clCreateContext(nullptr, 1, &candidate.device, nullptr, nullptr, &error);
    check(error, "create context for " + info_.name);
    const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, candidate.device, properties, &error);
    check(error, "create queue for " + info_.name);
    const std::string source = kernel_source();
    const char* source_ptr = source.c_str();
    const std::size_t source_size = source.size();
    program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_size, &error);
    check(error, "create program for " + info_.name);
    error = clBuildProgram(program_, 1, &candidate.device, "-cl-std=CL1.2", nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::size_t size{};
      clGetProgramBuildInfo(program_, candidate.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
      std::string log(size, '\0');
      clGetProgramBuildInfo(program_, candidate.device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
      throw std::runtime_error("build kernels for " + info_.name + " failed:\n" + log);
    }
    gray_ = create_kernel("rgb_to_gray");
    gaussian_ = create_kernel("gaussian3x3");
    downsample_ = create_kernel("downsample2");
    match_ = create_kernel("block_match");
  }

  ~Worker() {
    if (match_) clReleaseKernel(match_);
    if (downsample_) clReleaseKernel(downsample_);
    if (gaussian_) clReleaseKernel(gaussian_);
    if (gray_) clReleaseKernel(gray_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
  }

  const OpenClDeviceInfo& device_info() const { return info_; }

  std::array<cv::Mat, 4> preprocess(const cv::Mat& image) {
    const int width = image.cols;
    const int height = image.rows;
    const std::size_t pixels = image.total();
    cl_int error{};
    std::vector<cl_mem> allocations;
    auto create = [&](std::size_t bytes, cl_mem_flags flags, void* host = nullptr) {
      cl_mem value = clCreateBuffer(context_, flags, bytes, host, &error);
      check(error, "allocate preprocessing validation buffer on " + info_.name);
      allocations.push_back(value);
      return value;
    };
    try {
      cl_mem rgb = create(image.total() * image.elemSize(),
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image.data);
      cl_mem gray = create(pixels * sizeof(float), CL_MEM_READ_WRITE);
      cl_mem blur = create(pixels * sizeof(float), CL_MEM_READ_WRITE);
      const int width1 = std::max(1, width / 2);
      const int height1 = std::max(1, height / 2);
      const int width2 = std::max(1, width1 / 2);
      const int height2 = std::max(1, height1 / 2);
      cl_mem level1 = create(static_cast<std::size_t>(width1) * height1 * sizeof(float), CL_MEM_READ_WRITE);
      cl_mem level2 = create(static_cast<std::size_t>(width2) * height2 * sizeof(float), CL_MEM_READ_WRITE);
      run_gray(rgb, gray, width, height);
      run_gaussian(gray, blur, width, height);
      run_downsample(blur, level1, width, height);
      run_downsample(level1, level2, width1, height1);
      check(clFinish(queue_), "finish preprocessing validation on " + info_.name);
      std::array<cv::Mat, 4> output{cv::Mat(height, width, CV_32F), cv::Mat(height, width, CV_32F),
                                    cv::Mat(height1, width1, CV_32F), cv::Mat(height2, width2, CV_32F)};
      for (std::size_t i = 0; i < output.size(); ++i) {
        cl_mem source = std::array<cl_mem, 4>{gray, blur, level1, level2}[i];
        check(clEnqueueReadBuffer(queue_, source, CL_TRUE, 0, output[i].total() * sizeof(float),
                                  output[i].data, 0, nullptr, nullptr),
              "read preprocessing validation output from " + info_.name);
      }
      for (auto allocation : allocations) clReleaseMemObject(allocation);
      return output;
    } catch (...) {
      for (auto allocation : allocations) clReleaseMemObject(allocation);
      throw;
    }
  }

  WorkerResult run(const cv::Mat& previous, const cv::Mat& current, int grid_width,
                   int grid_height, int row_begin, int row_end) {
    if (row_begin >= row_end) return {{}, row_begin, row_end, 0.0};
    const auto started = Clock::now();
    cl_int error{};
    const int width = current.cols;
    const int height = current.rows;
    const std::size_t rgb_bytes = current.total() * current.elemSize();
    const std::size_t pixels = current.total();
    const std::size_t flow_count = static_cast<std::size_t>(grid_width) * grid_height;

    auto buffer = [&](cl_mem_flags flags, std::size_t bytes, void* host = nullptr) {
      cl_mem value = clCreateBuffer(context_, flags, bytes, host, &error);
      check(error, "allocate OpenCL buffer on " + info_.name);
      return value;
    };
    std::vector<cl_mem> allocations;
    auto owned = [&](cl_mem value) { allocations.push_back(value); return value; };
    try {
      cl_mem prev_rgb = owned(buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rgb_bytes, previous.data));
      cl_mem curr_rgb = owned(buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rgb_bytes, current.data));
      cl_mem prev_gray = owned(buffer(CL_MEM_READ_WRITE, pixels * sizeof(float)));
      cl_mem curr_gray = owned(buffer(CL_MEM_READ_WRITE, pixels * sizeof(float)));
      cl_mem prev_blur = owned(buffer(CL_MEM_READ_WRITE, pixels * sizeof(float)));
      cl_mem curr_blur = owned(buffer(CL_MEM_READ_WRITE, pixels * sizeof(float)));

      run_gray(prev_rgb, prev_gray, width, height);
      run_gray(curr_rgb, curr_gray, width, height);
      run_gaussian(prev_gray, prev_blur, width, height);
      run_gaussian(curr_gray, curr_blur, width, height);

      std::array<cl_mem, 3> prev_levels{prev_blur, nullptr, nullptr};
      std::array<cl_mem, 3> curr_levels{curr_blur, nullptr, nullptr};
      std::array<int, 3> widths{width, std::max(1, width / 2), std::max(1, width / 4)};
      std::array<int, 3> heights{height, std::max(1, height / 2), std::max(1, height / 4)};
      for (int level = 1; level < 3; ++level) {
        const std::size_t level_pixels = static_cast<std::size_t>(widths[level]) * heights[level];
        prev_levels[level] = owned(buffer(CL_MEM_READ_WRITE, level_pixels * sizeof(float)));
        curr_levels[level] = owned(buffer(CL_MEM_READ_WRITE, level_pixels * sizeof(float)));
        run_downsample(prev_levels[level - 1], prev_levels[level], widths[level - 1], heights[level - 1]);
        run_downsample(curr_levels[level - 1], curr_levels[level], widths[level - 1], heights[level - 1]);
      }

      std::vector<cl_float4> zero(flow_count);
      cl_mem prior = owned(buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  flow_count * sizeof(cl_float4), zero.data()));
      cl_mem output = owned(buffer(CL_MEM_READ_WRITE, flow_count * sizeof(cl_float4)));
      for (int level = 2; level >= 0; --level) {
        run_match(prev_levels[level], curr_levels[level], prior, output, widths[level],
                  heights[level], grid_width, row_begin, row_end, level);
        std::swap(prior, output);
      }
      check(clFinish(queue_), "finish perception on " + info_.name);
      std::vector<cl_float4> result(flow_count);
      check(clEnqueueReadBuffer(queue_, prior, CL_TRUE, 0, flow_count * sizeof(cl_float4),
                                result.data(), 0, nullptr, nullptr),
            "read flow from " + info_.name);
      for (auto allocation : allocations) clReleaseMemObject(allocation);
      const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
      return {std::move(result), row_begin, row_end, elapsed};
    } catch (...) {
      for (auto allocation : allocations) clReleaseMemObject(allocation);
      throw;
    }
  }

 private:
  cl_kernel create_kernel(const char* name) {
    cl_int error{};
    cl_kernel kernel = clCreateKernel(program_, name, &error);
    check(error, std::string("create kernel ") + name + " for " + info_.name);
    return kernel;
  }

  void enqueue_2d(cl_kernel kernel, std::size_t width, std::size_t height,
                  const std::string& operation) {
    const std::size_t global[] = {width, height};
    check(clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
          operation + " on " + info_.name);
  }

  void run_gray(cl_mem input, cl_mem output, int width, int height) {
    check(clSetKernelArg(gray_, 0, sizeof(input), &input), "set grayscale input");
    check(clSetKernelArg(gray_, 1, sizeof(output), &output), "set grayscale output");
    check(clSetKernelArg(gray_, 2, sizeof(width), &width), "set grayscale width");
    check(clSetKernelArg(gray_, 3, sizeof(height), &height), "set grayscale height");
    enqueue_2d(gray_, width, height, "grayscale");
  }

  void run_gaussian(cl_mem input, cl_mem output, int width, int height) {
    check(clSetKernelArg(gaussian_, 0, sizeof(input), &input), "set Gaussian input");
    check(clSetKernelArg(gaussian_, 1, sizeof(output), &output), "set Gaussian output");
    check(clSetKernelArg(gaussian_, 2, sizeof(width), &width), "set Gaussian width");
    check(clSetKernelArg(gaussian_, 3, sizeof(height), &height), "set Gaussian height");
    enqueue_2d(gaussian_, width, height, "Gaussian filter");
  }

  void run_downsample(cl_mem input, cl_mem output, int width, int height) {
    check(clSetKernelArg(downsample_, 0, sizeof(input), &input), "set pyramid input");
    check(clSetKernelArg(downsample_, 1, sizeof(output), &output), "set pyramid output");
    check(clSetKernelArg(downsample_, 2, sizeof(width), &width), "set pyramid width");
    check(clSetKernelArg(downsample_, 3, sizeof(height), &height), "set pyramid height");
    enqueue_2d(downsample_, std::max(1, width / 2), std::max(1, height / 2), "pyramid downsample");
  }

  void run_match(cl_mem previous, cl_mem current, cl_mem prior, cl_mem output,
                 int width, int height, int grid_width, int row_begin, int row_end, int level) {
    check(clSetKernelArg(match_, 0, sizeof(previous), &previous), "set matcher previous");
    check(clSetKernelArg(match_, 1, sizeof(current), &current), "set matcher current");
    check(clSetKernelArg(match_, 2, sizeof(prior), &prior), "set matcher prior");
    check(clSetKernelArg(match_, 3, sizeof(output), &output), "set matcher output");
    check(clSetKernelArg(match_, 4, sizeof(width), &width), "set matcher width");
    check(clSetKernelArg(match_, 5, sizeof(height), &height), "set matcher height");
    check(clSetKernelArg(match_, 6, sizeof(grid_width), &grid_width), "set matcher grid width");
    check(clSetKernelArg(match_, 7, sizeof(row_begin), &row_begin), "set matcher row begin");
    check(clSetKernelArg(match_, 8, sizeof(row_end), &row_end), "set matcher row end");
    check(clSetKernelArg(match_, 9, sizeof(level), &level), "set matcher level");
    enqueue_2d(match_, grid_width, row_end - row_begin, "block matching");
  }

  Candidate candidate_;
  OpenClDeviceInfo info_;
  cl_context context_{};
  cl_command_queue queue_{};
  cl_program program_{};
  cl_kernel gray_{};
  cl_kernel gaussian_{};
  cl_kernel downsample_{};
  cl_kernel match_{};
};

void fit_motion(PerceptionResult& result, cv::Size size) {
  std::vector<std::array<double, 7>> rows;
  for (const auto& flow : result.vectors) {
    if (flow.confidence < 0.12F) continue;
    const double w = std::sqrt(flow.confidence);
    rows.push_back({flow.origin.x * w, flow.origin.y * w, w, 0, 0, 0,
                    flow.displacement.x * w});
    rows.push_back({0, 0, 0, flow.origin.x * w, flow.origin.y * w, w,
                    flow.displacement.y * w});
  }
  result.focus_of_expansion = {-1.0F, -1.0F};
  if (rows.size() < 12) return;
  cv::Mat a(static_cast<int>(rows.size()), 6, CV_64F);
  cv::Mat b(static_cast<int>(rows.size()), 1, CV_64F);
  for (int r = 0; r < a.rows; ++r) {
    for (int c = 0; c < 6; ++c) a.at<double>(r, c) = rows[r][c];
    b.at<double>(r) = rows[r][6];
  }
  cv::Mat coefficients;
  if (!cv::solve(a, b, coefficients, cv::DECOMP_SVD)) return;
  const double ax = coefficients.at<double>(0);
  const double ay = coefficients.at<double>(1);
  const double bx = coefficients.at<double>(3);
  const double by = coefficients.at<double>(4);
  result.expansion = static_cast<float>((ax + by) * 0.5);
  cv::Mat linear = (cv::Mat_<double>(2, 2) << ax, ay, bx, by);
  cv::Mat translation = (cv::Mat_<double>(2, 1) << -coefficients.at<double>(2),
                          -coefficients.at<double>(5));
  cv::Mat foe;
  if (std::abs(cv::determinant(linear)) > 1e-7 && cv::solve(linear, translation, foe)) {
    const float x = static_cast<float>(foe.at<double>(0));
    const float y = static_cast<float>(foe.at<double>(1));
    if (x >= -size.width && x <= size.width * 2 && y >= -size.height && y <= size.height * 2) {
      result.focus_of_expansion = {x, y};
    }
  }
}

}  // namespace

std::vector<OpenClDeviceInfo> discover_opencl_devices() {
  std::vector<OpenClDeviceInfo> result;
  for (const auto& candidate : candidates()) result.push_back(info(candidate));
  return result;
}

KernelValidation validate_opencl_preprocessing() {
  const auto available = candidates();
  const auto candidate = std::find_if(available.begin(), available.end(),
                                      [](const Candidate& value) {
                                        return (value.type & CL_DEVICE_TYPE_CPU) != 0;
                                      });
  if (candidate == available.end()) {
    throw std::runtime_error("preprocessing validation requires an OpenCL CPU device");
  }
  cv::Mat input(37, 53, CV_8UC3);
  cv::RNG rng(0x51A1A4);
  rng.fill(input, cv::RNG::UNIFORM, 0, 255);
  Worker worker(*candidate);
  const auto actual = worker.preprocess(input);

  cv::Mat gray(input.rows, input.cols, CV_32F);
  for (int y = 0; y < input.rows; ++y) {
    for (int x = 0; x < input.cols; ++x) {
      const auto pixel = input.at<cv::Vec3b>(y, x);
      gray.at<float>(y, x) = 0.114F * pixel[0] + 0.587F * pixel[1] + 0.299F * pixel[2];
    }
  }
  cv::Mat gaussian;
  const cv::Mat weights = (cv::Mat_<float>(1, 3) << 0.25F, 0.5F, 0.25F);
  cv::sepFilter2D(gray, gaussian, CV_32F, weights, weights, {-1, -1}, 0.0, cv::BORDER_REPLICATE);
  const auto downsample = [](const cv::Mat& source) {
    cv::Mat output(std::max(1, source.rows / 2), std::max(1, source.cols / 2), CV_32F);
    for (int y = 0; y < output.rows; ++y) {
      for (int x = 0; x < output.cols; ++x) {
        const int x0 = std::min(x * 2, source.cols - 1);
        const int y0 = std::min(y * 2, source.rows - 1);
        const int x1 = std::min(x0 + 1, source.cols - 1);
        const int y1 = std::min(y0 + 1, source.rows - 1);
        output.at<float>(y, x) = 0.25F * (source.at<float>(y0, x0) + source.at<float>(y0, x1) +
                                          source.at<float>(y1, x0) + source.at<float>(y1, x1));
      }
    }
    return output;
  };
  const cv::Mat level1 = downsample(gaussian);
  const cv::Mat level2 = downsample(level1);
  const auto error = [](const cv::Mat& expected, const cv::Mat& observed) {
    return cv::norm(expected, observed, cv::NORM_INF);
  };
  return {error(gray, actual[0]), error(gaussian, actual[1]),
          error(level1, actual[2]), error(level2, actual[3])};
}

class OpenClPerception::Impl {
 public:
  explicit Impl(ExecutionMode mode) : mode(mode) {
    const auto available = candidates();
    auto find = [&](cl_device_type type) -> std::optional<Candidate> {
      const auto it = std::find_if(available.begin(), available.end(),
                                   [=](const Candidate& value) { return (value.type & type) != 0; });
      return it == available.end() ? std::nullopt : std::optional<Candidate>(*it);
    };
    if (mode == ExecutionMode::Cpu || mode == ExecutionMode::Fixed || mode == ExecutionMode::Adaptive) {
      auto value = find(CL_DEVICE_TYPE_CPU);
      if (!value) throw std::runtime_error("requested mode requires an OpenCL CPU device, but none was found");
      cpu = std::make_unique<Worker>(*value);
      device_infos.push_back(cpu->device_info());
    }
    if (mode == ExecutionMode::Gpu || mode == ExecutionMode::Fixed || mode == ExecutionMode::Adaptive) {
      auto value = find(CL_DEVICE_TYPE_GPU);
      if (!value) throw std::runtime_error("requested mode requires an OpenCL GPU device, but none was found");
      gpu = std::make_unique<Worker>(*value);
      device_infos.push_back(gpu->device_info());
    }
  }

  PerceptionResult process(const cv::Mat& previous, const cv::Mat& current, float ratio) {
    if (previous.empty() || current.empty() || previous.size() != current.size() ||
        previous.type() != CV_8UC3 || current.type() != CV_8UC3) {
      throw std::invalid_argument("perception requires equal non-empty CV_8UC3 frames");
    }
    const int grid_width = (current.cols + 15) / 16;
    const int grid_height = (current.rows + 15) / 16;
    int split = mode == ExecutionMode::Cpu ? grid_height
              : mode == ExecutionMode::Gpu ? 0
              : std::clamp(static_cast<int>(std::lround(grid_height * (1.0F - ratio))), 1,
                           grid_height - 1);
    WorkerResult cpu_result;
    WorkerResult gpu_result;
    const auto start = Clock::now();
    if (cpu && gpu) {
      auto cpu_future = std::async(std::launch::async, [&] {
        return cpu->run(previous, current, grid_width, grid_height, 0, split);
      });
      gpu_result = gpu->run(previous, current, grid_width, grid_height, split, grid_height);
      cpu_result = cpu_future.get();
    } else if (cpu) {
      cpu_result = cpu->run(previous, current, grid_width, grid_height, 0, grid_height);
    } else {
      gpu_result = gpu->run(previous, current, grid_width, grid_height, 0, grid_height);
    }

    PerceptionResult result;
    result.cpu_ms = cpu_result.elapsed_ms;
    result.gpu_ms = gpu_result.elapsed_ms;
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    result.vectors.reserve(static_cast<std::size_t>(grid_width) * grid_height);
    for (int y = 0; y < grid_height; ++y) {
      const auto& source = y < split && cpu ? cpu_result.flow : gpu_result.flow;
      for (int x = 0; x < grid_width; ++x) {
        const auto& raw = source[static_cast<std::size_t>(y) * grid_width + x];
        FlowVector flow;
        flow.origin = {static_cast<float>(std::min(x * 16 + 8, current.cols - 1)),
                       static_cast<float>(std::min(y * 16 + 8, current.rows - 1))};
        flow.displacement = {raw.s[0], raw.s[1]};
        flow.confidence = raw.s[2];
        const cv::Point2f radial = flow.origin - cv::Point2f(current.cols * 0.5F, current.rows * 0.5F);
        const float radius2 = radial.dot(radial);
        const float expansion = radius2 > 64.0F ? flow.displacement.dot(radial) / radius2 : 0.0F;
        if (expansion > 1e-5F) flow.ttc_proxy_s = 1.0F / (expansion * 30.0F);
        result.vectors.push_back(flow);
      }
    }
    fit_motion(result, current.size());
    return result;
  }

  ExecutionMode mode;
  std::unique_ptr<Worker> cpu;
  std::unique_ptr<Worker> gpu;
  std::vector<OpenClDeviceInfo> device_infos;
};

OpenClPerception::OpenClPerception(ExecutionMode mode) : impl_(std::make_unique<Impl>(mode)) {}
OpenClPerception::~OpenClPerception() = default;
OpenClPerception::OpenClPerception(OpenClPerception&&) noexcept = default;
OpenClPerception& OpenClPerception::operator=(OpenClPerception&&) noexcept = default;

PerceptionResult OpenClPerception::process(const cv::Mat& previous_bgr,
                                           const cv::Mat& current_bgr,
                                           float gpu_ratio) {
  return impl_->process(previous_bgr, current_bgr, gpu_ratio);
}

const std::vector<OpenClDeviceInfo>& OpenClPerception::devices() const {
  return impl_->device_infos;
}

}  // namespace flowguard
