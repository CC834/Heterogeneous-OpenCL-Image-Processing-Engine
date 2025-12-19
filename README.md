# Lab 6: Heterogeneous Programming in OpenCL

## Overview
This project implements heterogeneous image processing using OpenCL, utilizing both CPU and GPU devices concurrently to apply Gaussian blur to a stream of 5000 images.

## Hardware Tested
- CPU: Intel Core i7-12700
- GPU: Intel UHD Graphics 770 (integrated)

## Two Approaches Implemented

### Approach 1: Image-Level Distribution (`heterogeneous_blur.c`)
Complete images are assigned to either CPU or GPU based on a configurable ratio. Devices work independently on separate images.

### Approach 2: Split-Image Distribution (`split_image_blur.c`)
Each image is split horizontally between devices. CPU processes top rows, GPU processes bottom rows, with 1-pixel halo overlap for correct blur at boundaries.

## Files
```
├── heterogeneous_blur.c    # Approach 1: Image-level distribution
├── split_image_blur.c      # Approach 2: Split-image distribution
├── gaussian_kernel.cl      # OpenCL kernel (shared by both approaches)
├── gaussian_blur.c         # Single-device baseline (from Lab 5)
└── README.md               # This file
```

## Requirements
- OpenCL runtime (Intel OpenCL for CPU and GPU)
- CImg library (included in `./CImg/`)
- libjpeg

## Building
```
g++ -o split_image_blur split_image_blur.c -lOpenCL -ljpeg -lpthread
g++ -o heterogeneous_blur heterogeneous_blur.c -lOpenCL -lX11 -lpthread -lpng -ljpeg -I./CImg

```

## Usage

### Approach 1: Image-Level Distribution
```bash
# Heterogeneous mode (CPU + GPU)
./heterogeneous_blur both [gpu_ratio] [batch_size]

# CPU-only baseline
./heterogeneous_blur cpu

# GPU-only baseline
./heterogeneous_blur gpu

# Examples with optimal parameters
./heterogeneous_blur both 0.728 35    # Best performance
./heterogeneous_blur both 0.834 1200  # Best load balance
```

### Approach 2: Split-Image Distribution
```bash
./split_image_blur [gpu_ratio] [batch_size]

# Example with optimal parameters
./split_image_blur 0.837 35
```

## Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpu_ratio` | Fraction of work assigned to GPU (0.0-1.0) | 0.5 |
| `batch_size` | Images processed per batch | 500 |

## Optimal Configurations

### Approach 1 (Image-Level)
- **Best throughput:** batch=35, gpu_ratio=0.728 → 583ms, 8568 img/s
- **Best balance:** batch=1200, gpu_ratio=0.834 → 0% imbalance

### Approach 2 (Split-Image)
- **Best throughput:** batch=35, gpu_ratio=0.837 → 808ms, 6189 img/s
- **Best balance:** batch=35, gpu_ratio=0.837 → 0.3% imbalance

## Key Findings
1. Smaller batch sizes (35-50) significantly outperform larger ones (~2x speedup)
2. Approach 1 outperforms Approach 2 by ~1.38x for this workload
3. Heterogeneous execution beats GPU-only by 1.58x at optimal configuration
4. CPU becomes communication-bound at large batch sizes (Transfer OUT dominates)

## Ratio Calibration
To find optimal GPU ratio for your hardware:
1. Run with 50/50 split: `./heterogeneous_blur both 0.5 35`
2. Note the recommended ratio in output
3. Re-run with recommended ratio

Formula: `GPU_ratio = T_cpu / (T_cpu + T_gpu)`
