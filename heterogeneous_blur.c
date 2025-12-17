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

int main(int argc, char** argv)
{

    // Configuration
    int mode = 0;  // Default: heterogeneous
    const char *input_filename = "./image_320x240.jpg";
    const int NUM_IMAGES = 5000;  // Number of images in stream
    int local_work_size = 16;
    float gpu_ratio = 0.5f;  // Default: 50% to GPU (used in heterogeneous mode)
    
    
    // Execution mode: 0 = both (heterogeneous), 1 = CPU only, 2 = GPU only
    if (argc > 1) {
        if (strcmp(argv[1], "cpu") == 0) {
            mode = 1;
            printf("Mode: CPU ONLY\n");
        } else if (strcmp(argv[1], "gpu") == 0) {
            mode = 2;
            printf("Mode: GPU ONLY\n");
        } else if (strcmp(argv[1], "both") == 0) {
            mode = 0;
            printf("Mode: HETEROGENEOUS (CPU + GPU)\n");
        } else {
            printf("Usage: %s [cpu|gpu|both]\n", argv[0]);
            printf("Defaulting to heterogeneous mode.\n");
        }
    } else {
        printf("Mode: HETEROGENEOUS (CPU + GPU) [default]\n");
    }
    // Parse GPU ratio (optional second argument)
    if (argc > 2) {
        gpu_ratio = atof(argv[2]);
        if (gpu_ratio < 0.0f || gpu_ratio > 1.0f) {
            printf("Warning: gpu_ratio must be between 0.0 and 1.0. Using 0.5\n");
            gpu_ratio = 0.5f;
        }
    }

    // Only show ratio for heterogeneous mode
    if (mode == 0) {
        printf("GPU ratio: %.1f%% GPU, %.1f%% CPU\n", gpu_ratio * 100, (1 - gpu_ratio) * 100);
    }
    printf("========== HETEROGENEOUS CONFIGURATION ==========\n");
    printf("Input file: %s\n", input_filename);
    printf("Number of images in stream: %d\n", NUM_IMAGES);
    printf("Work-group size: %dx%d\n", local_work_size, local_work_size);
    printf("Execution mode : %d\n",mode);
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


    // ======================== CREATE IMAGE STREAM ========================

    printf("Creating image stream (%d images)...\n", NUM_IMAGES);

    // Allocate array of pointers
    unsigned char **image_stream = (unsigned char**)malloc(NUM_IMAGES * sizeof(unsigned char*));
    unsigned char **output_stream = (unsigned char**)malloc(NUM_IMAGES * sizeof(unsigned char*));

    if(!image_stream || !output_stream){printf("Error: Failed to allocate stream arrays\n");return -1;}

    // Allocate and copy each image
    for(int i = 0; i < NUM_IMAGES; i++){
        // Input: copy from original
        image_stream[i] = (unsigned char*)malloc(image_size * sizeof(unsigned char));
        memcpy(image_stream[i], original_image, image_size);
        
        // Output: just allocate (will be filled by devices)
        output_stream[i] = (unsigned char*)malloc(image_size * sizeof(unsigned char));
        
        if(!image_stream[i] || !output_stream[i]){
            printf("Error: Failed to allocate image %d\n", i);
            return -1;
        }
    }

    //size_t total_size_mb = (NUM_IMAGES * image_size * 2) / (1024.0 * 1024.0);  // *2 for input+output
    double total_size_mb = (double)(NUM_IMAGES * image_size * 2) / (1024.0 * 1024.0);
    printf("Image stream created: %d images\n", NUM_IMAGES);
    printf("Total memory allocated: %.2f MB\n", total_size_mb);
    printf("\n");


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
    
    //cl_command_queue q_cpu = clCreateCommandQueueWithProperties(ctx_cpu, device_cpu, 0, &err);
    //cl_error(err, "Failed to create CPU queue");

    //cl_command_queue q_gpu = clCreateCommandQueueWithProperties(ctx_gpu, device_gpu, 0, &err);
    //cl_error(err, "Failed to create GPU queue");

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
    source_code[source_size] = '\0';
    fread(source_code, 1, source_size, fp);
    fclose(fp);

    printf("Kernel source loaded (%zu bytes)\n", source_size);

    // Build program for CPU
    printf("Building kernel for CPU...\n");
    cl_program program_cpu = clCreateProgramWithSource(ctx_cpu, 1, 
                            (const char**)&source_code, &source_size, &err);
    cl_error(err, "Failed to create CPU program");

    err = clBuildProgram(program_cpu, 1, &device_cpu, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG, 
                              0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG, 
                              log_size, log, NULL);
        printf("CPU Build Error:\n%s\n", log);
        free(log);
        return -1;
    }
    printf("CPU kernel built successfully\n");

    // Build program for GPU
    printf("Building kernel for GPU...\n");
    cl_program program_gpu = clCreateProgramWithSource(ctx_gpu, 1, 
                            (const char**)&source_code, &source_size, &err);
    cl_error(err, "Failed to create GPU program");

    err = clBuildProgram(program_gpu, 1, &device_gpu, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG, 
                              0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG, 
                              log_size, log, NULL);
        printf("GPU Build Error:\n%s\n", log);
        free(log);
        return -1;
    }
    printf("GPU kernel built successfully\n");

    free(source_code);

    // Create kernel objects
    cl_kernel kernel_cpu = clCreateKernel(program_cpu, "gaussian_blur", &err);
    cl_error(err, "Failed to create CPU kernel");

    cl_kernel kernel_gpu = clCreateKernel(program_gpu, "gaussian_blur", &err);
    cl_error(err, "Failed to create GPU kernel");

    printf("Kernel objects created\n\n");




    // ======================== WORK DISTRIBUTION ========================
    // Calculate how many images each device will process based on mode
    int num_images_cpu = 0;
    int num_images_gpu = 0;

    if (mode == 0) {
        // Heterogeneous: split between both devices
        //num_images_cpu = NUM_IMAGES / 2;
        //num_images_gpu = NUM_IMAGES - num_images_cpu;  // GPU gets remainder
        // Heterogeneous: split based on gpu_ratio

        num_images_gpu = (int)(NUM_IMAGES * gpu_ratio);
        num_images_cpu = NUM_IMAGES - num_images_gpu;
    } else if (mode == 1) {
        // CPU only
        num_images_cpu = NUM_IMAGES;
        num_images_gpu = 0;
    } else if (mode == 2) {
        // GPU only
        num_images_cpu = 0;
        num_images_gpu = NUM_IMAGES;
    }
    
    // // Calculate how many images each device will process
    // int num_images_cpu = NUM_IMAGES / 2;  // Even indices: 0,2,4,...
    // int num_images_gpu = NUM_IMAGES / 2;  // Odd indices: 1,3,5,...

    // // If NUM_IMAGES is odd, CPU gets the extra one
    // if(NUM_IMAGES % 2 != 0) {
    //     num_images_cpu++;
    // }

    printf("Work distribution:\n");
    //printf("  CPU will process %d images (indices: 0,2,4,...)\n", num_images_cpu);
    //printf("  GPU will process %d images (indices: 1,3,5,...)\n", num_images_gpu);
    printf("  CPU will process %d images (indices: 0 to %d)\n", num_images_cpu, num_images_cpu - 1);
    printf("  GPU will process %d images (indices: %d to %d)\n", num_images_gpu, num_images_cpu, NUM_IMAGES - 1);
    printf("Total images: %d\n\n", num_images_cpu + num_images_gpu);
    
    // ======================== DEVICE BUFFER ALLOCATION ========================

    printf("Allocating device buffers...\n");

    // CPU buffers (for ONE image at a time)
    cl_mem buffer_cpu_input = clCreateBuffer(ctx_cpu, CL_MEM_READ_ONLY,
                                            image_size, NULL, &err);
    cl_error(err, "Failed to create CPU input buffer");

    cl_mem buffer_cpu_output = clCreateBuffer(ctx_cpu, CL_MEM_WRITE_ONLY,
                                              image_size, NULL, &err);
    cl_error(err, "Failed to create CPU output buffer");

    // GPU buffers (for ONE image at a time)
    cl_mem buffer_gpu_input = clCreateBuffer(ctx_gpu, CL_MEM_READ_ONLY,
                                            image_size, NULL, &err);
    cl_error(err, "Failed to create GPU input buffer");

    cl_mem buffer_gpu_output = clCreateBuffer(ctx_gpu, CL_MEM_WRITE_ONLY,
                                              image_size, NULL, &err);
    cl_error(err, "Failed to create GPU output buffer");

    printf("Device buffers allocated\n\n");

    // ======================== SET KERNEL ARGUMENTS ========================

    printf("Setting kernel arguments...\n");

    // CPU kernel arguments
    err = clSetKernelArg(kernel_cpu, 0, sizeof(cl_mem), &buffer_cpu_input);
    cl_error(err, "Failed to set CPU kernel arg 0");
    err = clSetKernelArg(kernel_cpu, 1, sizeof(cl_mem), &buffer_cpu_output);
    cl_error(err, "Failed to set CPU kernel arg 1");
    err = clSetKernelArg(kernel_cpu, 2, sizeof(int), &width);
    cl_error(err, "Failed to set CPU kernel arg 2");
    err = clSetKernelArg(kernel_cpu, 3, sizeof(int), &height);
    cl_error(err, "Failed to set CPU kernel arg 3");
    err = clSetKernelArg(kernel_cpu, 4, sizeof(int), &channels);
    cl_error(err, "Failed to set CPU kernel arg 4");

    // GPU kernel arguments
    err = clSetKernelArg(kernel_gpu, 0, sizeof(cl_mem), &buffer_gpu_input);
    cl_error(err, "Failed to set GPU kernel arg 0");
    err = clSetKernelArg(kernel_gpu, 1, sizeof(cl_mem), &buffer_gpu_output);
    cl_error(err, "Failed to set GPU kernel arg 1");
    err = clSetKernelArg(kernel_gpu, 2, sizeof(int), &width);
    cl_error(err, "Failed to set GPU kernel arg 2");
    err = clSetKernelArg(kernel_gpu, 3, sizeof(int), &height);
    cl_error(err, "Failed to set GPU kernel arg 3");
    err = clSetKernelArg(kernel_gpu, 4, sizeof(int), &channels);
    cl_error(err, "Failed to set GPU kernel arg 4");

    printf("Kernel arguments set\n\n");

    // ======================== WORK SIZE CALCULATION ========================

    size_t local_size[2] = {16, 16};
    size_t global_size[2];
    global_size[0] = ((width + local_size[0] - 1) / local_size[0]) * local_size[0];
    global_size[1] = ((height + local_size[1] - 1) / local_size[1]) * local_size[1];

    printf("Global work size: %zu x %zu\n", global_size[0], global_size[1]);
    printf("Local work size: %zu x %zu\n\n", local_size[0], local_size[1]); 
    
    
    // ======================== CONCURRENT PROCESSING ========================

    printf("Starting concurrent processing of %d images...\n", NUM_IMAGES);
    printf("Press Ctrl+C if this takes too long (we can optimize later)\n\n");

    // Timing variables
    double time_cpu_transfer_in = 0, time_cpu_kernel = 0, time_cpu_transfer_out = 0;
    double time_gpu_transfer_in = 0, time_gpu_kernel = 0, time_gpu_transfer_out = 0;

    double time_start_processing = get_time_ms();

    // Arrays to store events for all operations
    cl_event *cpu_events = (cl_event*)malloc(num_images_cpu * 3 * sizeof(cl_event)); // 3 events per image (write, kernel, read)
    cl_event *gpu_events = (cl_event*)malloc(num_images_gpu * 3 * sizeof(cl_event));

    int cpu_event_idx = 0;
    int gpu_event_idx = 0;

    // Launch ALL work asynchronously (non-blocking)
    for(int img_idx = 0; img_idx < NUM_IMAGES; img_idx++) {
        
        if(img_idx % 500 == 0) {
            printf("Launching image %d/%d...\n", img_idx, NUM_IMAGES);
        }
        
        // Decide which device processes this image
        int use_cpu = 0;

        if (mode == 1) {
            // CPU only mode
            use_cpu = 1;
        } else if (mode == 2) {
            // GPU only mode
            use_cpu = 0;
        } else {
            // Heterogeneous mode: first num_images_cpu go to CPU, rest to GPU
            use_cpu = (img_idx < num_images_cpu);
        }
        
        if (use_cpu) {
            // ============ CPU processes this image ============
            
            // Transfer input to CPU (NON-BLOCKING!)
            err = clEnqueueWriteBuffer(q_cpu, buffer_cpu_input, CL_FALSE, 0,
                                      image_size, image_stream[img_idx],
                                      0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU write failed");
            
            // Launch kernel on CPU (NON-BLOCKING!)
            err = clEnqueueNDRangeKernel(q_cpu, kernel_cpu, 2, NULL,
                                        global_size, local_size,
                                        0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU kernel launch failed");
            
            // Read result from CPU (NON-BLOCKING!)
            err = clEnqueueReadBuffer(q_cpu, buffer_cpu_output, CL_FALSE, 0,
                                      image_size, output_stream[img_idx],
                                      0, NULL, &cpu_events[cpu_event_idx++]);
            cl_error(err, "CPU read failed");
            
        } else {
            // ============ GPU processes this image ============
            
            // Transfer input to GPU (NON-BLOCKING!)
            err = clEnqueueWriteBuffer(q_gpu, buffer_gpu_input, CL_FALSE, 0,
                                      image_size, image_stream[img_idx],
                                      0, NULL, &gpu_events[gpu_event_idx++]);
            cl_error(err, "GPU write failed");
            
            // Launch kernel on GPU (NON-BLOCKING!)
            err = clEnqueueNDRangeKernel(q_gpu, kernel_gpu, 2, NULL,
                                        global_size, local_size,
                                        0, NULL, &gpu_events[gpu_event_idx++]);
            cl_error(err, "GPU kernel launch failed");
            
            // Read result from GPU (NON-BLOCKING!)
            err = clEnqueueReadBuffer(q_gpu, buffer_gpu_output, CL_FALSE, 0,
                                      image_size, output_stream[img_idx],
                                      0, NULL, &gpu_events[gpu_event_idx++]);
            cl_error(err, "GPU read failed");
        }
}

    printf("All work queued! Waiting for completion...\n");

    // NOW wait for both devices to finish ALL their work
    if (num_images_cpu > 0) clFinish(q_cpu);
    if (num_images_gpu > 0) clFinish(q_gpu);

    double time_end_processing = get_time_ms();
    double time_total_processing = time_end_processing - time_start_processing;

    printf("All devices finished!\n\n");

    // ======================== COLLECT TIMING FROM EVENTS ========================

    printf("Collecting timing data...\n");

    // Process CPU events
    if (num_images_cpu > 0) {
    for(int i = 0; i < cpu_event_idx; i += 3) {
        cl_ulong t_start, t_end;
        
        // Write event
        clGetEventProfilingInfo(cpu_events[i], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(cpu_events[i], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_cpu_transfer_in += (t_end - t_start) / 1000000.0;
        
        // Kernel event
        clGetEventProfilingInfo(cpu_events[i+1], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(cpu_events[i+1], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_cpu_kernel += (t_end - t_start) / 1000000.0;
        
        // Read event
        clGetEventProfilingInfo(cpu_events[i+2], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(cpu_events[i+2], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_cpu_transfer_out += (t_end - t_start) / 1000000.0;
    }
    }

    // Process GPU events
    if (num_images_gpu > 0) {
    for(int i = 0; i < gpu_event_idx; i += 3) {
        cl_ulong t_start, t_end;
        
        // Write event
        clGetEventProfilingInfo(gpu_events[i], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(gpu_events[i], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_gpu_transfer_in += (t_end - t_start) / 1000000.0;
        
        // Kernel event
        clGetEventProfilingInfo(gpu_events[i+1], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(gpu_events[i+1], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_gpu_kernel += (t_end - t_start) / 1000000.0;
        
        // Read event
        clGetEventProfilingInfo(gpu_events[i+2], CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(gpu_events[i+2], CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        time_gpu_transfer_out += (t_end - t_start) / 1000000.0;
    }
    }

    // Cleanup events
    for(int i = 0; i < cpu_event_idx; i++) {
        clReleaseEvent(cpu_events[i]);
    }
    for(int i = 0; i < gpu_event_idx; i++) {
        clReleaseEvent(gpu_events[i]);
    }

    free(cpu_events);
    free(gpu_events);

    printf("Timing data collected!\n\n");


    // ======================== PERFORMANCE ANALYSIS ========================

    printf("========== PERFORMANCE RESULTS ==========\n\n");

    // Overall execution time
    printf("1. OVERALL EXECUTION TIME\n");
    printf("   Total wall-clock time: %.2f ms (%.2f seconds)\n", 
          time_total_processing, time_total_processing/1000.0);
    printf("\n");

    // CPU breakdown
    double time_cpu_total;
    if (num_images_cpu > 0) {
    double time_cpu_total = time_cpu_transfer_in + time_cpu_kernel + time_cpu_transfer_out;
    printf("2. CPU DEVICE (processed %d images)\n", num_images_cpu);
    printf("   Total CPU time:        %.2f ms\n", time_cpu_total);
    printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", 
          time_cpu_transfer_in, (time_cpu_transfer_in/time_cpu_total)*100);
    printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", 
          time_cpu_kernel, (time_cpu_kernel/time_cpu_total)*100);
    printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", 
          time_cpu_transfer_out, (time_cpu_transfer_out/time_cpu_total)*100);
    printf("   Average per image:     %.2f ms\n", time_cpu_total / num_images_cpu);
    printf("\n");
    }

    // GPU breakdown
    double time_gpu_total;
    if (num_images_gpu > 0) {
    double time_gpu_total = time_gpu_transfer_in + time_gpu_kernel + time_gpu_transfer_out;
    printf("3. GPU DEVICE (processed %d images)\n", num_images_gpu);
    printf("   Total GPU time:        %.2f ms\n", time_gpu_total);
    printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", 
          time_gpu_transfer_in, (time_gpu_transfer_in/time_gpu_total)*100);
    printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", 
          time_gpu_kernel, (time_gpu_kernel/time_gpu_total)*100);
    printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", 
          time_gpu_transfer_out, (time_gpu_transfer_out/time_gpu_total)*100);
    printf("   Average per image:     %.2f ms\n", time_gpu_total / num_images_gpu);
    printf("\n");
    }
    printf("====================");

    if (num_images_cpu > 0 && num_images_gpu > 0) {
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
    // ======================== CLEANUP ========================

    //printf("Cleaning up resources...\n");

    // Release OpenCL buffers
    clReleaseMemObject(buffer_cpu_input);
    clReleaseMemObject(buffer_cpu_output);
    clReleaseMemObject(buffer_gpu_input);
    clReleaseMemObject(buffer_gpu_output);

    // Release kernels
    clReleaseKernel(kernel_cpu);
    clReleaseKernel(kernel_gpu);

    // Release programs
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

    for (int i = 0; i < NUM_IMAGES; i++) {
        free(image_stream[i]);
        free(output_stream[i]);
    }

    free(image_stream);
    free(output_stream);

    //printf("Cleanup complete!\n");
    return 0;
}