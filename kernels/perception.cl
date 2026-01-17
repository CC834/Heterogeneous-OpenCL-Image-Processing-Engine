__kernel void rgb_to_gray(__global const uchar* bgr, __global float* gray,
                          const int width, const int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;
  const int i = (y * width + x) * 3;
  gray[y * width + x] = 0.114f * bgr[i] + 0.587f * bgr[i + 1] + 0.299f * bgr[i + 2];
}

__kernel void gaussian3x3(__global const float* input, __global float* output,
                          const int width, const int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;
  const float w[3] = {0.25f, 0.5f, 0.25f};
  float sum = 0.0f;
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      const int px = clamp(x + i, 0, width - 1);
      const int py = clamp(y + j, 0, height - 1);
      sum += input[py * width + px] * w[i + 1] * w[j + 1];
    }
  }
  output[y * width + x] = sum;
}

__kernel void downsample2(__global const float* input, __global float* output,
                          const int input_width, const int input_height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int width = max(1, input_width / 2);
  const int height = max(1, input_height / 2);
  if (x >= width || y >= height) return;
  const int x0 = min(x * 2, input_width - 1);
  const int y0 = min(y * 2, input_height - 1);
  const int x1 = min(x0 + 1, input_width - 1);
  const int y1 = min(y0 + 1, input_height - 1);
  output[y * width + x] = 0.25f * (input[y0 * input_width + x0] +
                                   input[y0 * input_width + x1] +
                                   input[y1 * input_width + x0] +
                                   input[y1 * input_width + x1]);
}

__kernel void block_match(__global const float* previous,
                          __global const float* current,
                          __global const float4* prior,
                          __global float4* flow,
                          const int width, const int height,
                          const int grid_width, const int row_begin,
                          const int row_end, const int level) {
  const int tile_x = get_global_id(0);
  const int local_row = get_global_id(1);
  const int tile_y = row_begin + local_row;
  if (tile_x >= grid_width || tile_y >= row_end) return;
  const int out_index = tile_y * grid_width + tile_x;
  const int scale = 1 << level;
  const int cx = clamp((tile_x * 16 + 8) / scale, 0, width - 1);
  const int cy = clamp((tile_y * 16 + 8) / scale, 0, height - 1);
  const int radius = max(2, 8 / scale);
  const float4 p = prior[out_index];
  const int predicted_x = level == 2 ? 0 : (int)round(p.x * 2.0f);
  const int predicted_y = level == 2 ? 0 : (int)round(p.y * 2.0f);
  float best = FLT_MAX;
  float second = FLT_MAX;
  int best_x = predicted_x;
  int best_y = predicted_y;

  for (int dy = predicted_y - 6; dy <= predicted_y + 6; ++dy) {
    for (int dx = predicted_x - 6; dx <= predicted_x + 6; ++dx) {
      float sad = 0.0f;
      int samples = 0;
      for (int by = -radius; by < radius; by += 2) {
        for (int bx = -radius; bx < radius; bx += 2) {
          const int px = cx + bx;
          const int py = cy + by;
          const int qx = px + dx;
          const int qy = py + dy;
          if (px >= 0 && py >= 0 && px < width && py < height &&
              qx >= 0 && qy >= 0 && qx < width && qy < height) {
            sad += fabs(previous[py * width + px] - current[qy * width + qx]);
            ++samples;
          }
        }
      }
      if (samples == 0) continue;
      sad /= samples;
      if (sad < best) {
        second = best;
        best = sad;
        best_x = dx;
        best_y = dy;
      } else if (sad < second) {
        second = sad;
      }
    }
  }
  const float confidence = isfinite(second) ? clamp((second - best) / (second + 1e-4f), 0.0f, 1.0f) : 0.0f;
  flow[out_index] = (float4)((float)best_x, (float)best_y, confidence, best);
}
