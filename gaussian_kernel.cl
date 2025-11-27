
// Host side (CPU):

// Load image with CImg
// Extract RGB data to arrays
// Create OpenCL setup (context, queue, etc.)
// Copy image data to device
// Launch kernel
// Get blurred image back
// Display/save result

// Device side (GPU) - The Kernel:

// Each work-item processes ONE pixel
// Read 3x3 neighborhood
// Apply Gaussian weights
// Write result

__kernel void gaussian_blur(
    __global const unsigned char * input, // Input image (RGB)
    __global unsigned char *output, // Output the blurred image
    const int width,                 // width of the image
    const int height,               // Height of the image
    const int channels              // 3 channels for RGB
)
{

    // Get pixel coordinates 
    int x = get_global_id(0); // Column
    int y = get_global_id(1); // Row

    // Boundary check
    if(x >= width || y >= height) return;

    // Gussian 3x3 kernal weights
    float gaussian_kernel[3][3] =
    {
        {0.0625f, 0.125f, 0.0625f},
        {0.125f,  0.25f,  0.125f},
        {0.0625f, 0.125f, 0.0625f}
    };

    // Process each color channel (R,G,B)
    for (int c = 0; c < channels; c++){
        float sum = 0.0f;

        // Apply 3x3
        for(int ky = -1; ky <= 1; ky++){      // Kernel rows
            for(int kx = -1; kx <= 1; kx++){  // Kernel columns

                // Calculate neighbor coordinates
                int nx = x + kx;
                int ny = y + ky;

                // Handle boundaries (clamp to edge)
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
            
                // Calculate array index: (y * width + x) * channels + channel
                int index = (ny * width + nx) * channels + c;

                // Multiply pixel value by kernel weight and accumulate
                sum += input[index] * gaussian_kernel[ky + 1][kx + 1];  

            }
        }

        // Write result to output
        int out_index = (y * width + x) * channels + c;
        output[out_index] = (unsigned char)sum;
    }
}