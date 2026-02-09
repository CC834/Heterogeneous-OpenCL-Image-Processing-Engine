#include "test_support.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "flowguard/opencl_pipeline.hpp"

using namespace flowguard;
using flowguard::test::require;

namespace {

cv::Mat textured_frame() {
  cv::Mat image(181, 319, CV_8UC3);
  cv::RNG rng(0xF10A6A4D);
  rng.fill(image, cv::RNG::UNIFORM, 0, 255);
  return image;
}

std::vector<float> confident_dx(const PerceptionResult& result) {
  std::vector<float> values;
  for (const auto& vector : result.vectors) {
    if (vector.confidence >= 0.2F) values.push_back(vector.displacement.x);
  }
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

TEST_CASE("OpenCL grayscale Gaussian and pyramid kernels match scalar references") {
  const auto errors = validate_opencl_preprocessing();
  require(errors.grayscale_max_error <= 1e-3, "grayscale kernel exceeded 1e-3 tolerance");
  require(errors.gaussian_max_error <= 1e-3, "Gaussian kernel exceeded 1e-3 tolerance");
  require(errors.pyramid_level1_max_error <= 1e-3, "pyramid level 1 exceeded 1e-3 tolerance");
  require(errors.pyramid_level2_max_error <= 1e-3, "pyramid level 2 exceeded 1e-3 tolerance");
}

TEST_CASE("OpenCL pyramid flow preserves zero motion on non-divisible dimensions") {
  const cv::Mat image = textured_frame();
  OpenClPerception pipeline(ExecutionMode::Cpu);
  const auto result = pipeline.process(image, image, 0.0F);
  const auto dx = confident_dx(result);
  require(dx.size() > result.vectors.size() / 2, "too few confident zero-motion blocks");
  for (const auto& vector : result.vectors) {
    if (vector.confidence >= 0.2F) {
      require(cv::norm(vector.displacement) < 0.01F, "zero-motion block moved");
    }
  }
}

TEST_CASE("OpenCL three-level block flow recovers deterministic translation") {
  const cv::Mat previous = textured_frame();
  cv::Mat current;
  const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 2, 0, 1, 0);
  cv::warpAffine(previous, current, transform, previous.size(), cv::INTER_NEAREST,
                 cv::BORDER_REPLICATE);
  OpenClPerception pipeline(ExecutionMode::Cpu);
  const auto dx = confident_dx(pipeline.process(previous, current, 0.0F));
  require(!dx.empty(), "translation produced no confident blocks");
  require(std::abs(dx[dx.size() / 2] - 2.0F) <= 0.5F, "median translation differs from scalar fixture");
}

TEST_CASE("CPU and GPU modes produce equivalent block vectors when GPU exists") {
  const auto devices = discover_opencl_devices();
  const bool has_gpu = std::any_of(devices.begin(), devices.end(),
                                   [](const auto& device) { return device.type == "gpu"; });
  if (!has_gpu) return;  // CI's explicit PoCL-only environment has no GPU.
  const cv::Mat previous = textured_frame();
  cv::Mat current;
  const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 1);
  cv::warpAffine(previous, current, transform,
                 previous.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
  OpenClPerception cpu(ExecutionMode::Cpu);
  OpenClPerception gpu(ExecutionMode::Gpu);
  const auto cpu_result = cpu.process(previous, current, 0.0F);
  const auto gpu_result = gpu.process(previous, current, 1.0F);
  require(cpu_result.vectors.size() == gpu_result.vectors.size(), "mode grid sizes differ");
  for (std::size_t i = 0; i < cpu_result.vectors.size(); ++i) {
    require(cv::norm(cpu_result.vectors[i].displacement - gpu_result.vectors[i].displacement) <= 0.01F,
            "CPU/GPU displacement exceeded 0.01 pixel tolerance");
    require(std::abs(cpu_result.vectors[i].confidence - gpu_result.vectors[i].confidence) <= 1e-4F,
            "CPU/GPU confidence exceeded 1e-4 tolerance");
  }
}
