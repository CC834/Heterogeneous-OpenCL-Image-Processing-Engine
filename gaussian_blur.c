////////////////////////////////////////////////////////////////////
//File: basic_environ.c
//
//Description: base file for environment exercises with openCL
//
// 
////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>  // For gettimeofday()

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#define cimg_use_jpeg
#include "./CImg/CImg.h"
using namespace cimg_library;

// ABOUT ERRORS
// Remmember to check error codes from every OpenCL API call
// Info about error codes can be found at the reference manual of OpenCL
// At the following url, you can find name for each error code
//  https://gist.github.com/bmount/4a7144ce801e5569a0b6
//  https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
// Following function checks errors, and in such a case, it prints the code, the string and it exits

void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

// Helper function to get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}
////////////////////////////////////////////////////////////////////////////////


/* ATTENTION: While prgramming in OpenCL it is a good idea to keep the reference manuals handy:
 * https://bashbaug.github.io/OpenCL-Docs/pdf/OpenCL_API.pdf
 * https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf (summary of OpenCL API)
 * https://www.khronos.org/assets/uploads/developers/presentations/opencl20-quick-reference-card.pdf
 */

#include "opencl_utils.h"

int main(int argc, char** argv)
{
  // START TIMING - Overall program
  double time_start_total = get_time_ms();
  
  // Parse command-line arguments
  const char *input_filename = "input.jpg";
  int local_work_size = 16;
  
  if (argc >= 2) {
      input_filename = argv[1];
  }
  if (argc >= 3) {
      local_work_size = atoi(argv[2]);
  }
  
  printf("========== CONFIGURATION ==========\n");
  printf("Input file: %s\n", input_filename);
  printf("Work-group size: %dx%d\n", local_work_size, local_work_size);
  printf("===================================\n\n");

  int err;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;


// ======================== LOAD IMAGE ========================f

// printf("Loading image...\n");
CImg<unsigned char> img(input_filename);

int width = img.width();
int height = img.height();
int channels = img.spectrum();

printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

// Calculate total size
size_t image_size = width * height * channels;

// Allocate host memory
unsigned char *host_input = (unsigned char*)malloc(image_size * sizeof(unsigned char));
unsigned char *host_output = (unsigned char*)malloc(image_size * sizeof(unsigned char));

if(!host_input || !host_output){
    printf("Error: Failed to allocate host memory\n");
    return -1;
}

// ============== setup openCL ==============================

if (scan_platforms_and_devices(&platform_id, &device_id) != 0) {
    printf("Error: No valid OpenCL platform/device found\n");return -1;
}

if (create_context_and_queue(platform_id, device_id, &context, &command_queue) != 0) {
    printf("Error: at function create context and queue\n");return -1;
}



// Convert CImg (planar) to interleaved (RGBRGBRGB...)
// printf("Converting image format...\n");
for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
        int idx = (y * width + x) * channels;
        host_input[idx + 0] = img(x, y, 0, 0);  // Red
        host_input[idx + 1] = img(x, y, 0, 1);  // Green
        host_input[idx + 2] = img(x, y, 0, 2);  // Blue
    }
}

// printf("Image data prepared!\n\n");
// ============== LOAD TO KERNAL =============================

//const unsigned int count = 1024;  // easy to tweak later
//cl_program program;float *host_input  = NULL;float *host_output = NULL;
//if (build_program_and_host_buffers(context,device_id,"kernel.cl",count,&program,&host_input,&host_output) != 0) {return -1;}
// ============== LOAD AND BUILD KERNEL =============================

// printf("Loading kernel source...\n");

FILE *fp = fopen("gaussian_kernel.cl", "r");
if(!fp){
    printf("Error: Cannot open gaussian_kernel.cl\n");
    free(host_input);
    free(host_output);
    return -1;
}

// Get file size
fseek(fp, 0, SEEK_END);
size_t source_size = ftell(fp);
rewind(fp);

// Read source code
char *source_code = (char*)malloc(source_size + 1);
source_code[source_size] = '\0';
fread(source_code, 1, source_size, fp);
fclose(fp);

// Create program
cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_code, 
                                                &source_size, &err);
cl_error(err, "Failed to create program");
free(source_code);

// Build program
err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
if(err != CL_SUCCESS){
    // Print build log if there's an error
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char*)malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("Build Error:\n%s\n", log);
    free(log);
    return -1;
}

// printf("Kernel built successfully!\n\n");

// ========================Create The Kernal Object ========================

//cl_kernel kernel;cl_mem device_input, device_output;
//if (setup_kernel_and_buffers(context,command_queue,program,"pow_of_two",count,host_input,&kernel,&device_input,&device_output) != 0) {return -1;}

// printf("Creating kernel and buffers...\n");

// Create kernel
cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &err);
cl_error(err, "Failed to create kernel");

// Create device buffers
cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                    image_size * sizeof(unsigned char), NULL, &err);
cl_error(err, "Failed to create input buffer");

cl_mem device_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    image_size * sizeof(unsigned char), NULL, &err);
cl_error(err, "Failed to create output buffer");

// Copy input image to device
// err = clEnqueueWriteBuffer(command_queue, device_input, CL_TRUE, 0,
//                             image_size * sizeof(unsigned char), host_input,
//                             0, NULL, NULL);
// cl_error(err, "Failed to write input to device");

// Measure Host to Device transfer
  cl_event write_event;
  double time_write_start = get_time_ms();
  
  err = clEnqueueWriteBuffer(command_queue, device_input, CL_TRUE, 0,
                             image_size * sizeof(unsigned char), host_input,
                             0, NULL, &write_event);
  cl_error(err, "Failed to write input to device");
  
  double time_write_end = get_time_ms();
  double time_write = time_write_end - time_write_start;
  
  // Calculate bandwidth (MB/s)
  double data_size_mb = (image_size * sizeof(unsigned char)) / (1024.0 * 1024.0);
  double bandwidth_write = data_size_mb / (time_write / 1000.0); // MB/s
  
//   printf("Host->Device transfer time: %.2f ms (%.2f MB/s)\n", 
//          time_write, bandwidth_write);
// printf("Buffers created and input copied to device!\n\n");

// ================ Set kernal arugments ================

// printf("Setting kernel arguments...\n");

// Argument 0: input image buffer

err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input);
cl_error(err, "Failed to set arg 0");

// Argument 1: output image buffer
err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_output);
cl_error(err, "Failed to set arg 1");

// Argument 2: width
err = clSetKernelArg(kernel, 2, sizeof(int), &width);
cl_error(err, "Failed to set arg 2");

// Argument 3: height
err = clSetKernelArg(kernel, 3, sizeof(int), &height);
cl_error(err, "Failed to set arg 3");

// Argument 4: channels
err = clSetKernelArg(kernel, 4, sizeof(int), &channels);
cl_error(err, "Failed to set arg 4");

// printf("All kernel arguments set!\n\n");

// ============== LAUNCH KERNEL (2D!) =============================
  
  // printf("Launching kernel...\n");
  
  // Define 2D work sizes
  //size_t local_size[2] = {16, 16};   // 16x16 = 256 work-items per work-group
  size_t local_size[2] = {(size_t)local_work_size, (size_t)local_work_size};

  // Global size must be multiple of local size
  // Round up width and height to nearest multiple of 16
  size_t global_size[2];
  global_size[0] = ((width + local_size[0] - 1) / local_size[0]) * local_size[0];   // Width
  global_size[1] = ((height + local_size[1] - 1) / local_size[1]) * local_size[1];  // Height
  
  printf("Global size: %zu x %zu\n", global_size[0], global_size[1]);
  printf("Local size: %zu x %zu\n", local_size[0], local_size[1]);
  printf("Total work-items: %zu\n", global_size[0] * global_size[1]);
  
  // Launch with 2D (second parameter is 2, not 1!)
  //   err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
  //                                global_size, local_size, 0, NULL, NULL);
  // cl_error(err, "Failed to launch kernel");

  // Create event for profiling kernel execution
  cl_event kernel_event;
  
  // Launch with 2D (second parameter is 2, not 1!)
  err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                               global_size, local_size, 
                               0, NULL, &kernel_event);  // Pass event here
  cl_error(err, "Failed to launch kernel");

  // Wait for kernel to finish
  clFinish(command_queue);

  // printf("Kernel execution complete!\n\n");
  
  // Get kernel execution time from event
  cl_ulong time_start_kernel, time_end_kernel;
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, 
                          sizeof(time_start_kernel), &time_start_kernel, NULL);
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, 
                          sizeof(time_end_kernel), &time_end_kernel, NULL);
  
  double kernel_time_ms = (time_end_kernel - time_start_kernel) / 1000000.0; // Convert nanoseconds to ms
  
  //printf("2. Kernel execution time: %.2f ms\n", kernel_time_ms);
  
  // Calculate throughput (pixels per second)
  int total_pixels = width * height;
  double throughput = (total_pixels / (kernel_time_ms / 1000.0)) / 1000000.0; // Megapixels/sec
  
  //printf("4. Kernel throughput: %.2f Mpixels/sec\n", throughput);

// ==============0 Read result from device to host ==============

// printf("Reading results from device...\n");

// err = clEnqueueReadBuffer(command_queue, device_output, CL_TRUE, 0,
//                             image_size * sizeof(unsigned char), host_output,
//                             0, NULL, NULL);
// cl_error(err, "Failed to read output from device");

cl_event read_event;
double time_read_start = get_time_ms();

err = clEnqueueReadBuffer(command_queue, device_output, CL_TRUE, 0,
                        image_size * sizeof(unsigned char), host_output,
                        0, NULL, &read_event);
cl_error(err, "Failed to read output from device");

double time_read_end = get_time_ms();
double time_read = time_read_end - time_read_start;

double bandwidth_read = data_size_mb / (time_read / 1000.0); // MB/s

//printf("Device->Host transfer time: %.2f ms (%.2f MB/s)\n", 
//        time_read, bandwidth_read);

// printf("Results read successfully!\n\n");


// ============== SAVE OUTPUT IMAGE =============================

// printf("Converting output format and saving...\n");

// Create output CImg
CImg<unsigned char> output_img(width, height, 1, channels);

// Convert from interleaved back to planar
for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
        int idx = (y * width + x) * channels;
        output_img(x, y, 0, 0) = host_output[idx + 0];  // Red
        output_img(x, y, 0, 1) = host_output[idx + 1];  // Green
        output_img(x, y, 0, 2) = host_output[idx + 2];  // Blue
    }
}

// Save the blurred image
output_img.save("output_blurred.jpg");
printf("Saved to output_blurred.jpg\n\n");

// Display both images (optional - opens windows)
// img.display("Original");
// output_img.display("Blurred");

// ============== CLEANUP =============================

// printf("Cleaning up...\n");

clReleaseMemObject(device_input);
clReleaseMemObject(device_output);
clReleaseKernel(kernel);
clReleaseProgram(program);
clReleaseCommandQueue(command_queue);
clReleaseContext(context);

//  release the event at cleanup
clReleaseEvent(kernel_event);
clReleaseEvent(write_event);
clReleaseEvent(read_event);

free(host_input);
free(host_output);

// printf("Done!\n");

// END TIMING - Overall program
double time_end_total = get_time_ms();
double time_total = time_end_total - time_start_total;

printf("\n========== PERFORMANCE METRICS ==========\n");

// 1. Overall execution time
printf("1. Overall execution time: %.2f ms\n", time_total);

// 2. Kernel execution time
printf("2. Kernel execution time: %.2f ms\n", kernel_time_ms);

// 3. Memory bandwidth (group both transfers together)
printf("3. Memory bandwidth:\n");
printf("   Host->Device: %.2f ms (%.2f MB/s)\n", time_write, bandwidth_write);
printf("   Device->Host: %.2f ms (%.2f MB/s)\n", time_read, bandwidth_read);

// 4. Kernel throughput
printf("4. Kernel throughput: %.2f Mpixels/sec\n", throughput);

// 5. Memory footprint
printf("5. Memory footprint:\n");
printf("   Host input buffer:  %.2f MB\n", data_size_mb);
printf("   Host output buffer: %.2f MB\n", data_size_mb);
printf("   Device input buffer:  %.2f MB\n", data_size_mb);
printf("   Device output buffer: %.2f MB\n", data_size_mb);
printf("   Total memory used: %.2f MB\n", data_size_mb * 4);

printf("=========================================\n\n");
return 0;
}

