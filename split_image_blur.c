#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#define cimg_display 0
#define cimg_use_jpeg
#include "./CImg/CImg.h"

using namespace cimg_library;


// Helper functions (copy from Lab 5)
void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

static void save_one_image_rgb(const char* filename,
                               const unsigned char* interleaved,
                               int width, int height, int channels)
{
    CImg<unsigned char> out(width, height, 1, channels);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            out(x, y, 0, 0) = interleaved[idx + 0];
            out(x, y, 0, 1) = interleaved[idx + 1];
            out(x, y, 0, 2) = interleaved[idx + 2];
        }
    }

    out.save(filename); // extension decides format (.jpg/.png)
}


int main(int argc, char** argv)
{

    // Configuration
    const bool SAVE_IMAGE = false;
    const char *input_filename = "./image_320x240.jpg";
    const int NUM_IMAGES = 5000;  // Total number of images in stream
    int BATCH_SIZE = 500;   // Images per batch
    int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;  // Ceiling division
    int local_work_size = 16;
    float gpu_ratio = 0.5f;  // Default: 50% to GPU (used in heterogeneous mode)
    const int HALO = 1;  // Overlap rows needed for 3x3 Gaussian kernel
    

    // Parse GPU ratio (optional second argument)
    if (argc > 1) {
        gpu_ratio = atof(argv[1]);
        if (gpu_ratio < 0.0f || gpu_ratio > 1.0f) {
            printf("Warning: gpu_ratio must be between 0.0 and 1.0. Using 0.5\n");
            gpu_ratio = 0.5f;
        }
    }

    // Parse batch size (second argument)
    if (argc > 2) {
        BATCH_SIZE = atoi(argv[2]);
        if (BATCH_SIZE < 1 || BATCH_SIZE > NUM_IMAGES) {
            printf("Warning: BATCH_SIZE must be between 1 and %d. Using 500\n", NUM_IMAGES);
            BATCH_SIZE = 500;
        }
    }
    // ALWAYS recompute after final BATCH_SIZE is known
    NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

 
    printf("========== SPLIT-IMAGE CONFIGURATION ==========\n");
    printf("Input file: %s\n", input_filename);
    printf("Number of images in stream: %d\n", NUM_IMAGES);
    printf("Batch size: %d images\n", BATCH_SIZE);
    printf("Number of batches: %d\n", NUM_BATCHES);
    printf("Work-group size: %dx%d\n", local_work_size, local_work_size);
    printf("GPU ratio: %.1f%% (rows to GPU)\n", gpu_ratio * 100);
    printf("Halo size: %d row(s)\n", HALO);
    printf("================================================\n\n");
    


    // ======================== LOAD ORIGINAL IMAGE ========================

    CImg<unsigned char> img(input_filename);

    int width = img.width();
    int height = img.height();
    int channels = img.spectrum();

    printf("Original image loaded: %dx%d, %d channels\n", width, height, channels);

    // Calculate size of ONE image
    size_t image_size = width * height * channels;
    printf("Size of one image: %zu bytes (%.2f KB)\n", 
          image_size, image_size / 1024.0);

    // Allocate buffer for original image
    unsigned char *original_image = (unsigned char*)malloc(image_size * sizeof(unsigned char));

    if(!original_image){
        printf("Error: Failed to allocate memory for original image\n");
        return -1;
    }

    // Convert CImg (planar) to interleaved format (RGBRGBRGB...)
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int idx = (y * width + x) * channels;
            original_image[idx + 0] = img(x, y, 0, 0);  // Red
            original_image[idx + 1] = img(x, y, 0, 1);  // Green
            original_image[idx + 2] = img(x, y, 0, 2);  // Blue
        }
    }

    printf("Original image converted to interleaved format\n\n");


    // ======================== CALCULATE SPLIT DIMENSIONS ========================

    int split_row = (int)(height * (1.0f - gpu_ratio));  // CPU gets top rows

    // Safety check for edge cases
    if (split_row < HALO) {
        printf("Warning: split_row too small, adjusting to %d\n", HALO);
        split_row = HALO;
    }
    if (split_row > height - HALO) {
        printf("Warning: split_row too large, adjusting to %d\n", height - HALO);
        split_row = height - HALO;
    }

    // CPU dimensions (with halo)
    int cpu_input_rows = split_row + HALO;          // Extra row below for blur
    int cpu_output_rows = split_row;                 // Actual output rows
    size_t cpu_input_size = width * cpu_input_rows * channels;
    size_t cpu_output_size = width * cpu_output_rows * channels;

    // GPU dimensions (with halo)
    int gpu_input_rows = (height - split_row) + HALO;  // Extra row above for blur
    int gpu_output_rows = height - split_row;          // Actual output rows
    size_t gpu_input_size = width * gpu_input_rows * channels; // Input buffer
    size_t gpu_output_size = width * gpu_output_rows * channels; // Output Buffer

    printf("Split configuration:\n");
    printf("  Split row: %d (CPU: rows 0-%d, GPU: rows %d-%d)\n", split_row, split_row-1, split_row, height-1);
    printf("  CPU: %d input rows (inc. halo), %d output rows\n", cpu_input_rows, cpu_output_rows);
    printf("  GPU: %d input rows (inc. halo), %d output rows\n", gpu_input_rows, gpu_output_rows);
    printf("  CPU input size: %.2f KB, output size: %.2f KB\n", cpu_input_size/1024.0, cpu_output_size/1024.0);
    printf("  GPU input size: %.2f KB, output size: %.2f KB\n\n", gpu_input_size/1024.0, gpu_output_size/1024.0);

    // ======================== OPENCL DEVICE DISCOVERY ========================

    cl_int err;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_error(err, "Failed to count platforms");
    if (num_platforms == 0) { printf("No OpenCL platforms found\n"); return -1; }

    cl_platform_id *plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, plats, NULL);
    cl_error(err, "Failed to get platforms");

    cl_platform_id plat_cpu = NULL, plat_gpu = NULL;
    cl_device_id device_cpu = NULL, device_gpu = NULL;

    for (cl_uint p = 0; p < num_platforms; p++) {
        char pname[256] = {0};
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        printf("Platform %u: %s\n", p, pname);

        // Try GPU on this platform
        if (!device_gpu) {
            cl_device_id dev;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) {
                device_gpu = dev;
                plat_gpu = plats[p];
            }
        }

        // Try CPU on this platform
        if (!device_cpu) {
            cl_device_id dev;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, 1, &dev, NULL) == CL_SUCCESS) {
                device_cpu = dev;
                plat_cpu = plats[p];
            }
        }
    }

    free(plats);

    if (!device_cpu || !device_gpu) {
        printf("Error: Could not find both CPU and GPU devices\n");
        return -1;
    }

    char dname[256];
    clGetDeviceInfo(device_cpu, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("CPU device: %s\n", dname);

    clGetDeviceInfo(device_gpu, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("GPU device: %s\n", dname);

    // Create TWO contexts (one per platform)
    cl_context ctx_cpu = clCreateContext(NULL, 1, &device_cpu, NULL, NULL, &err);
    cl_error(err, "Failed to create CPU context");

    cl_context ctx_gpu = clCreateContext(NULL, 1, &device_gpu, NULL, NULL, &err);
    cl_error(err, "Failed to create GPU context");

    // Create TWO queues WITH PROFILING ENABLED
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,        // We want to set queue properties
        CL_QUEUE_PROFILING_ENABLE,  // Enable timing data collection
        0                           // Null terminator (end of list)
    };

    // Create TWO queues
    cl_command_queue q_cpu = clCreateCommandQueueWithProperties(ctx_cpu, device_cpu, props, &err);
    cl_error(err, "Failed to create CPU queue");

    cl_command_queue q_gpu = clCreateCommandQueueWithProperties(ctx_gpu, device_gpu, props, &err);
    cl_error(err, "Failed to create GPU queue");

    printf("\n");


    // ======================== LOAD AND BUILD KERNEL ========================

    printf("Loading kernel source...\n");

    // Read kernel source file
    FILE *fp = fopen("gaussian_kernel.cl", "r");
    if(!fp){
        printf("Error: Cannot open gaussian_kernel.cl\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);

    char *source_code = (char*)malloc(source_size + 1);
    if(!source_code){
        printf("Error: Failed to allocate kernel source buffer\n");
        fclose(fp);
        return -1;
    }
    source_code[source_size] = '\0';

    size_t read_bytes = fread(source_code, 1, source_size, fp);
    fclose(fp);

    if(read_bytes != source_size){
        printf("Warning: fread read %zu/%zu bytes\n", read_bytes, source_size);
    }

    printf("Kernel source loaded (%zu bytes)\n", source_size);

    // Build program(s) depending on mode
    cl_program program_cpu = NULL;
    cl_program program_gpu = NULL;

    cl_kernel kernel_cpu = NULL;
    cl_kernel kernel_gpu = NULL;

    // ---------------- CPU ----------------
    printf("Building kernel for CPU...\n");

    program_cpu = clCreateProgramWithSource(ctx_cpu, 1,
                    (const char**)&source_code, &source_size, &err);
    cl_error(err, "Failed to create CPU program");

    err = clBuildProgram(program_cpu, 1, &device_cpu, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t log_size = 0;
        clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);

        char *log = (char*)malloc(log_size + 1);
        if(log){
            log[log_size] = '\0';
            clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG,
                                    log_size, log, NULL);
            printf("CPU Build Error:\n%s\n", log);
            free(log);
        } else {
            printf("CPU Build Error (and failed to allocate log)\n");
        }
        free(source_code);
        return -1;
    }

    printf("CPU kernel built successfully\n");

    kernel_cpu = clCreateKernel(program_cpu, "gaussian_blur", &err);
    cl_error(err, "Failed to create CPU kernel");

    // ---------------- GPU ----------------
    printf("Building kernel for GPU...\n");

    program_gpu = clCreateProgramWithSource(ctx_gpu, 1,
                    (const char**)&source_code, &source_size, &err);
    cl_error(err, "Failed to create GPU program");

    err = clBuildProgram(program_gpu, 1, &device_gpu, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t log_size = 0;
        clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);

        char *log = (char*)malloc(log_size + 1);
        if(log){
            log[log_size] = '\0';
            clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG,
                                    log_size, log, NULL);
            printf("GPU Build Error:\n%s\n", log);
            free(log);
        } else {
            printf("GPU Build Error (and failed to allocate log)\n");
        }
        free(source_code);
        return -1;
    }

    printf("GPU kernel built successfully\n");

    kernel_gpu = clCreateKernel(program_gpu, "gaussian_blur", &err);
    cl_error(err, "Failed to create GPU kernel");

    // We can free source now
    free(source_code);

    printf("Kernel objects created\n\n");


    // ======================== DEVICE BUFFER ALLOCATION ========================

    printf("Allocating device buffers...\n");

    cl_mem buffer_cpu_input  = NULL;
    cl_mem buffer_cpu_output = NULL;
    cl_mem buffer_gpu_input  = NULL;
    cl_mem buffer_gpu_output = NULL;

    // CPU buffers 
    buffer_cpu_input = clCreateBuffer(ctx_cpu, CL_MEM_READ_ONLY, cpu_input_size, NULL, &err);
    cl_error(err, "Failed to create CPU input buffer");

    //buffer_cpu_output = clCreateBuffer(ctx_cpu, CL_MEM_WRITE_ONLY, cpu_output_size, NULL, &err);
    buffer_cpu_output = clCreateBuffer(ctx_cpu, CL_MEM_WRITE_ONLY, cpu_input_size, NULL, &err);
    cl_error(err, "Failed to create CPU output buffer");

    // GPU buffers 
    buffer_gpu_input = clCreateBuffer(ctx_gpu, CL_MEM_READ_ONLY, gpu_input_size, NULL, &err);
    cl_error(err, "Failed to create GPU input buffer");

    //buffer_gpu_output = clCreateBuffer(ctx_gpu, CL_MEM_WRITE_ONLY, gpu_output_size, NULL, &err);
    buffer_gpu_output = clCreateBuffer(ctx_gpu, CL_MEM_WRITE_ONLY, gpu_input_size, NULL, &err);
    cl_error(err, "Failed to create GPU output buffer");

    printf("Device buffers allocated\n\n");


    // ======================== SET KERNEL ARGUMENTS ========================

    printf("Setting kernel arguments...\n");

    // CPU kernel arguments (only if CPU is used)
    err = clSetKernelArg(kernel_cpu, 0, sizeof(cl_mem), &buffer_cpu_input);
    cl_error(err, "Failed to set CPU kernel arg 0");
    err = clSetKernelArg(kernel_cpu, 1, sizeof(cl_mem), &buffer_cpu_output);
    cl_error(err, "Failed to set CPU kernel arg 1");
    err = clSetKernelArg(kernel_cpu, 2, sizeof(int), &width);
    cl_error(err, "Failed to set CPU kernel arg 2");
    //err = clSetKernelArg(kernel_cpu, 3, sizeof(int), &height);
    err = clSetKernelArg(kernel_cpu, 3, sizeof(int), &cpu_input_rows); // The height of its own input buffer (including halo), not the full image height.
    cl_error(err, "Failed to set CPU kernel arg 3");
    err = clSetKernelArg(kernel_cpu, 4, sizeof(int), &channels);
    cl_error(err, "Failed to set CPU kernel arg 4");

    // GPU kernel arguments (only if GPU is used)
    err = clSetKernelArg(kernel_gpu, 0, sizeof(cl_mem), &buffer_gpu_input);
    cl_error(err, "Failed to set GPU kernel arg 0");
    err = clSetKernelArg(kernel_gpu, 1, sizeof(cl_mem), &buffer_gpu_output);
    cl_error(err, "Failed to set GPU kernel arg 1");
    err = clSetKernelArg(kernel_gpu, 2, sizeof(int), &width);
    cl_error(err, "Failed to set GPU kernel arg 2");
    //err = clSetKernelArg(kernel_gpu, 3, sizeof(int), &height);
    err = clSetKernelArg(kernel_gpu, 3, sizeof(int), &gpu_input_rows); // The height of its own input buffer (including halo), not the full image height.
    cl_error(err, "Failed to set GPU kernel arg 3");
    err = clSetKernelArg(kernel_gpu, 4, sizeof(int), &channels);
    cl_error(err, "Failed to set GPU kernel arg 4");

    printf("Kernel arguments set\n\n");


    // ======================== WORK SIZE CALCULATION ========================

    size_t local_size[2] = {16, 16};

    // CPU global size (based on cpu_input_rows)
    size_t global_size_cpu[2];
    global_size_cpu[0] = ((width + local_size[0] - 1) / local_size[0]) * local_size[0];
    global_size_cpu[1] = ((cpu_input_rows + local_size[1] - 1) / local_size[1]) * local_size[1];

    // GPU global size (based on gpu_input_rows)
    size_t global_size_gpu[2];
    global_size_gpu[0] = ((width + local_size[0] - 1) / local_size[0]) * local_size[0];
    global_size_gpu[1] = ((gpu_input_rows + local_size[1] - 1) / local_size[1]) * local_size[1];

    printf("CPU global size: %zu x %zu\n", global_size_cpu[0], global_size_cpu[1]);
    printf("GPU global size: %zu x %zu\n", global_size_gpu[0], global_size_gpu[1]);
    printf("Local size: %zu x %zu\n\n", local_size[0], local_size[1]);


    // ======================== BATCH PROCESSING ========================

    printf("Starting batch processing of %d images in %d batches...\n\n", NUM_IMAGES, NUM_BATCHES);

    // Total timing variables (accumulated across all batches)
    double time_cpu_transfer_in = 0, time_cpu_kernel = 0, time_cpu_transfer_out = 0;
    double time_gpu_transfer_in = 0, time_gpu_kernel = 0, time_gpu_transfer_out = 0;
    

    double time_start_total = get_time_ms();

    // ======================== BATCH LOOP ========================
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        
        printf("=== Processing Batch %d/%d ===\n", batch + 1, NUM_BATCHES);

        // Calculate actual batch size (handles leftover images in last batch)
        int batch_start = batch * BATCH_SIZE;
        if (batch_start >= NUM_IMAGES) break;
        int batch_count = BATCH_SIZE;
        if (batch_start + batch_count > NUM_IMAGES)
            batch_count = NUM_IMAGES - batch_start;
        if (batch_start + batch_count > NUM_IMAGES) {
            batch_count = NUM_IMAGES - batch_start;
        }

        // ======================== CREATE BATCH IMAGE STREAM (CONTIGUOUS) ========================

        unsigned char *batch_input = (unsigned char*)malloc(batch_count * image_size);
        unsigned char *batch_output = (unsigned char*)malloc(batch_count * image_size);

        if(!batch_input || !batch_output){
            printf("Error: Failed to allocate batch memory\n");
            return -1;
        }

        // Copy original image to each slot in contiguous buffer
        for(int i = 0; i < batch_count; i++){
            memcpy(batch_input + i * image_size, original_image, image_size);
        }

        // ======================== WORK DISTRIBUTION FOR THIS BATCH ========================

        // Every image is processed by BOTH devices
        printf("  Processing %d images (each split between CPU and GPU)\n", batch_count);

        // ======================== CONCURRENT PROCESSING ========================

        // Arrays to store events for this batch
        // Each image has 3 events per device: transfer in, kernel, transfer out
        cl_event *cpu_events = (cl_event*)malloc(batch_count * 3 * sizeof(cl_event));
        cl_event *gpu_events = (cl_event*)malloc(batch_count * 3 * sizeof(cl_event));

        if (!cpu_events || !gpu_events) {
            printf("Error: Failed to allocate event arrays\n");
            return -1;
        }

        int cpu_event_idx = 0;
        int gpu_event_idx = 0;

        // Launch ALL work asynchronously (non-blocking)
        for(int img_idx = 0; img_idx < batch_count; img_idx++) {
            
            // Pointer to this image in contiguous buffer
            unsigned char *in_ptr = batch_input + img_idx * image_size;
            unsigned char *out_ptr = batch_output + img_idx * image_size;
            
            // Calculate pointers for split regions
            // CPU: rows 0 to split_row-1, but INPUT includes halo (row split_row)
            unsigned char *cpu_in_ptr = in_ptr;  // Start at row 0
            unsigned char *cpu_out_ptr = out_ptr; // Output starts at row 0
            
            // GPU: rows split_row to height-1, but INPUT includes halo (row split_row-1)
            // GPU input starts one row BEFORE split_row (the halo row)
            unsigned char *gpu_in_ptr = in_ptr + (split_row - HALO) * width * channels;
            unsigned char *gpu_out_ptr = out_ptr + split_row * width * channels;
            
            // ============ CPU processes top portion ============
            err = clEnqueueWriteBuffer(q_cpu, buffer_cpu_input, CL_FALSE, 0,cpu_input_size, cpu_in_ptr,0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU write failed");
            
            err = clEnqueueNDRangeKernel(q_cpu, kernel_cpu, 2, NULL,global_size_cpu, local_size,0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU kernel launch failed");
            
            err = clEnqueueReadBuffer(q_cpu, buffer_cpu_output, CL_FALSE, 0,cpu_output_size, cpu_out_ptr,0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU read failed");
            
            // ============ GPU processes bottom portion ============
            err = clEnqueueWriteBuffer(q_gpu, buffer_gpu_input, CL_FALSE, 0,gpu_input_size, gpu_in_ptr,0, NULL, &gpu_events[gpu_event_idx++]);
            cl_error(err, "GPU write failed");
            
            err = clEnqueueNDRangeKernel(q_gpu, kernel_gpu, 2, NULL,global_size_gpu, local_size,0, NULL, &gpu_events[gpu_event_idx++]);
            cl_error(err, "GPU kernel launch failed");
            
            //err = clEnqueueReadBuffer(q_gpu, buffer_gpu_output, CL_FALSE, 0, gpu_output_size, gpu_out_ptr, 0, NULL, &gpu_events[gpu_event_idx++]);
            err = clEnqueueReadBuffer(q_gpu, buffer_gpu_output, CL_FALSE, HALO * width * channels,  // Skip first row (halo)
                          gpu_output_size, gpu_out_ptr, 
                          0, NULL, &gpu_events[gpu_event_idx++]);

            cl_error(err, "GPU read failed");
        }

        // Wait for this batch to complete
        clFinish(q_cpu);
        clFinish(q_gpu);

        if (SAVE_IMAGE == TRUE){
            if (batch == 0) {
                save_one_image_rgb("split_output.jpg", batch_output, width, height, channels);
                printf("Saved example output: split_output.jpg\n");
            }
        }

        // ======================== COLLECT TIMING FROM EVENTS ========================

        // Process CPU events
        for(int i = 0; i < cpu_event_idx; i += 3) {
            cl_ulong t_start, t_end;
            
            clGetEventProfilingInfo(cpu_events[i], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(cpu_events[i], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_cpu_transfer_in += (t_end - t_start) / 1000000.0;
            
            clGetEventProfilingInfo(cpu_events[i+1], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(cpu_events[i+1], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_cpu_kernel += (t_end - t_start) / 1000000.0;
            
            clGetEventProfilingInfo(cpu_events[i+2], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(cpu_events[i+2], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_cpu_transfer_out += (t_end - t_start) / 1000000.0;
        }

        // Process GPU events
        for(int i = 0; i < gpu_event_idx; i += 3) {
            cl_ulong t_start, t_end;
            
            clGetEventProfilingInfo(gpu_events[i], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(gpu_events[i], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_gpu_transfer_in += (t_end - t_start) / 1000000.0;
            
            clGetEventProfilingInfo(gpu_events[i+1], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(gpu_events[i+1], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_gpu_kernel += (t_end - t_start) / 1000000.0;
            
            clGetEventProfilingInfo(gpu_events[i+2], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
            clGetEventProfilingInfo(gpu_events[i+2], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
            time_gpu_transfer_out += (t_end - t_start) / 1000000.0;
        }

        // Cleanup events for this batch
        for(int i = 0; i < cpu_event_idx; i++) {
            clReleaseEvent(cpu_events[i]);
        }
        free(cpu_events);
        for(int i = 0; i < gpu_event_idx; i++) {
            clReleaseEvent(gpu_events[i]);
        }
        free(gpu_events);

        // Free batch memory (just 2 frees now!)
        free(batch_input);
        free(batch_output);

        printf("  Batch %d complete.\n\n", batch + 1);
    }
    // ======================== END BATCH LOOP ========================

    double time_end_total = get_time_ms();
    double time_total_processing = time_end_total - time_start_total;

    printf("All batches finished!\n\n");


    // ======================== PERFORMANCE ANALYSIS ========================

    printf("========== PERFORMANCE RESULTS ==========\n\n");

    // Overall execution time
    printf("1. OVERALL EXECUTION TIME\n");
    printf("   Total wall-clock time: %.2f ms (%.2f seconds)\n", 
          time_total_processing, time_total_processing/1000.0);
    printf("   Total images processed: %d\n", NUM_IMAGES);
    printf("\n");

    // CPU breakdown
    double time_cpu_total = time_cpu_transfer_in + time_cpu_kernel + time_cpu_transfer_out;
    printf("2. CPU DEVICE (processed %d images - top %d rows each)\n", NUM_IMAGES, cpu_output_rows);
    printf("   Total CPU time:        %.2f ms\n", time_cpu_total);
    printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", time_cpu_transfer_in, (time_cpu_transfer_in/time_cpu_total)*100);
    printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", time_cpu_kernel, (time_cpu_kernel/time_cpu_total)*100);
    printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", time_cpu_transfer_out, (time_cpu_transfer_out/time_cpu_total)*100);
    printf("\n");

    // GPU breakdown
    double time_gpu_total = time_gpu_transfer_in + time_gpu_kernel + time_gpu_transfer_out;
    printf("3. GPU DEVICE (processed %d images - bottom %d rows each)\n", NUM_IMAGES, gpu_output_rows);
    printf("   Total GPU time:        %.2f ms\n", time_gpu_total);
    printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", time_gpu_transfer_in, (time_gpu_transfer_in/time_gpu_total)*100);
    printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", time_gpu_kernel, (time_gpu_kernel/time_gpu_total)*100);
    printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", time_gpu_transfer_out, (time_gpu_transfer_out/time_gpu_total)*100);
    printf("\n");

    printf("============================\n");

        // Device comparison
        printf("4. DEVICE COMPARISON\n");
        double speedup_factor = time_cpu_total / time_gpu_total;
        if(speedup_factor > 1.0) {
            printf("   GPU is %.2fx FASTER than CPU\n", speedup_factor);
        } else {
            printf("   CPU is %.2fx FASTER than GPU\n", 1.0/speedup_factor);
        }
        printf("   CPU/GPU time ratio: %.2f\n", speedup_factor);
        printf("\n");

        // Workload balance
        printf("5. WORKLOAD BALANCE\n");
        double imbalance = fabs(time_cpu_total - time_gpu_total) / 
                          fmax(time_cpu_total, time_gpu_total) * 100.0;
        printf("   Workload imbalance: %.1f%%\n", imbalance);
        if(time_cpu_total > time_gpu_total) {
            printf("   CPU is the BOTTLENECK (%.2f ms slower)\n", 
                  time_cpu_total - time_gpu_total);
        } else {
            printf("   GPU is the BOTTLENECK (%.2f ms slower)\n", 
                  time_gpu_total - time_cpu_total);
        }
        printf("\n");

        // Bottleneck identification per device
        printf("6. BOTTLENECK IDENTIFICATION\n");
        printf("   CPU bottleneck: ");
        if(time_cpu_transfer_in + time_cpu_transfer_out > time_cpu_kernel) {
            printf("COMMUNICATION (%.1f%% of time)\n", 
                  ((time_cpu_transfer_in + time_cpu_transfer_out)/time_cpu_total)*100);
        } else {
            printf("COMPUTATION (%.1f%% of time)\n", 
                  (time_cpu_kernel/time_cpu_total)*100);
        }

        printf("   GPU bottleneck: ");
        if(time_gpu_transfer_in + time_gpu_transfer_out > time_gpu_kernel) {
            printf("COMMUNICATION (%.1f%% of time)\n", 
                  ((time_gpu_transfer_in + time_gpu_transfer_out)/time_gpu_total)*100);
        } else {
            printf("COMPUTATION (%.1f%% of time)\n", 
                  (time_gpu_kernel/time_gpu_total)*100);
        }
    printf("\n");

    // Throughput
    printf("7. THROUGHPUT\n");
    double throughput_mpixels = (NUM_IMAGES * width * height) / 
                                (time_total_processing / 1000.0) / 1000000.0;
    printf("   Overall throughput: %.2f Megapixels/sec\n", throughput_mpixels);
    printf("   Images per second: %.2f\n", 
          NUM_IMAGES / (time_total_processing / 1000.0));
    printf("\n");

    printf("=========================================\n\n");

    // Per-image timing and optimal ratio calculation
    printf("8. SPLIT-IMAGE STATISTICS\n");
    printf("   CPU time per image: %.3f ms (for %d rows)\n", time_cpu_total / NUM_IMAGES, cpu_output_rows);
    printf("   GPU time per image: %.3f ms (for %d rows)\n", time_gpu_total / NUM_IMAGES, gpu_output_rows);
    printf("   Combined time per image: %.3f ms\n", time_total_processing / NUM_IMAGES);
    printf("   Current GPU ratio: %.1f%%\n", gpu_ratio * 100);
    printf("\n");

    // Calculate optimal ratio based on time per row
    double cpu_time_per_row = time_cpu_total / (NUM_IMAGES * cpu_output_rows);
    double gpu_time_per_row = time_gpu_total / (NUM_IMAGES * gpu_output_rows);
    double optimal_gpu_ratio = cpu_time_per_row / (cpu_time_per_row + gpu_time_per_row);

    printf("9. OPTIMAL RATIO RECOMMENDATION\n");
    printf("   CPU: %.5f ms/row\n", cpu_time_per_row);
    printf("   GPU: %.5f ms/row\n", gpu_time_per_row);
    printf("   Recommended GPU ratio: %.1f%%\n", optimal_gpu_ratio * 100);
    printf("   Run with: ./split_image_blur %.3f\n", optimal_gpu_ratio);
    printf("\n");

    // ======================== CLEANUP ========================

    clReleaseMemObject(buffer_cpu_input);
    clReleaseMemObject(buffer_cpu_output);
    clReleaseMemObject(buffer_gpu_input);
    clReleaseMemObject(buffer_gpu_output);

    clReleaseKernel(kernel_cpu);
    clReleaseKernel(kernel_gpu);

    clReleaseProgram(program_cpu);
    clReleaseProgram(program_gpu);

    // Release command queues
    clReleaseCommandQueue(q_cpu);
    clReleaseCommandQueue(q_gpu);

    // Release contexts
    clReleaseContext(ctx_cpu);
    clReleaseContext(ctx_gpu);

    // Free host memory
    free(original_image);

    return 0;
}