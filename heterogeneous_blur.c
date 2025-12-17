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
    const int NUM_IMAGES = 5000;  // Total number of images in stream
    const int BATCH_SIZE = 500;   // Images per batch
    const int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;  // Ceiling division
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
    printf("Batch size: %d images\n", BATCH_SIZE);
    printf("Number of batches: %d\n", NUM_BATCHES);
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
    if (mode != 2) {
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
    }

    // ---------------- GPU ----------------
    if (mode != 1) {
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
    }

    // We can free source now
    free(source_code);

    printf("Kernel objects created\n\n");


    // ======================== DEVICE BUFFER ALLOCATION ========================

    printf("Allocating device buffers...\n");

    cl_mem buffer_cpu_input  = NULL;
    cl_mem buffer_cpu_output = NULL;
    cl_mem buffer_gpu_input  = NULL;
    cl_mem buffer_gpu_output = NULL;

    // CPU buffers (only if CPU is used)
    if (mode != 2) {
        buffer_cpu_input = clCreateBuffer(ctx_cpu, CL_MEM_READ_ONLY, image_size, NULL, &err);
        cl_error(err, "Failed to create CPU input buffer");

        buffer_cpu_output = clCreateBuffer(ctx_cpu, CL_MEM_WRITE_ONLY, image_size, NULL, &err);
        cl_error(err, "Failed to create CPU output buffer");
    }

    // GPU buffers (only if GPU is used)
    if (mode != 1) {
        buffer_gpu_input = clCreateBuffer(ctx_gpu, CL_MEM_READ_ONLY, image_size, NULL, &err);
        cl_error(err, "Failed to create GPU input buffer");

        buffer_gpu_output = clCreateBuffer(ctx_gpu, CL_MEM_WRITE_ONLY, image_size, NULL, &err);
        cl_error(err, "Failed to create GPU output buffer");
    }

    printf("Device buffers allocated\n\n");


    // ======================== SET KERNEL ARGUMENTS ========================

    printf("Setting kernel arguments...\n");

    // CPU kernel arguments (only if CPU is used)
    if (mode != 2) {
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
    }

    // GPU kernel arguments (only if GPU is used)
    if (mode != 1) {
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
    }

    printf("Kernel arguments set\n\n");


    // ======================== WORK SIZE CALCULATION ========================

    size_t local_size[2] = {16, 16};
    size_t global_size[2];
    global_size[0] = ((width + local_size[0] - 1) / local_size[0]) * local_size[0];
    global_size[1] = ((height + local_size[1] - 1) / local_size[1]) * local_size[1];

    printf("Global work size: %zu x %zu\n", global_size[0], global_size[1]);
    printf("Local work size: %zu x %zu\n\n", local_size[0], local_size[1]); 


    // ======================== BATCH PROCESSING ========================

    printf("Starting batch processing of %d images in %d batches...\n\n", NUM_IMAGES, NUM_BATCHES);

    // Total timing variables (accumulated across all batches)
    double time_cpu_transfer_in = 0, time_cpu_kernel = 0, time_cpu_transfer_out = 0;
    double time_gpu_transfer_in = 0, time_gpu_kernel = 0, time_gpu_transfer_out = 0;
    int total_images_cpu = 0, total_images_gpu = 0;

    double time_start_total = get_time_ms();

    // ======================== BATCH LOOP ========================
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        
        printf("=== Processing Batch %d/%d ===\n", batch + 1, NUM_BATCHES);

        // Calculate actual batch size (handles leftover images in last batch)
        int batch_start = batch * BATCH_SIZE;
        int batch_count = BATCH_SIZE;
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

        int num_images_cpu = 0;
        int num_images_gpu = 0;

        if (mode == 0) {
            num_images_gpu = (int)(batch_count * gpu_ratio);
            num_images_cpu = batch_count - num_images_gpu;
        } else if (mode == 1) {
            num_images_cpu = batch_count;
            num_images_gpu = 0;
        } else if (mode == 2) {
            num_images_cpu = 0;
            num_images_gpu = batch_count;
        }

        total_images_cpu += num_images_cpu;
        total_images_gpu += num_images_gpu;

        printf("  Batch work distribution: CPU=%d, GPU=%d\n", num_images_cpu, num_images_gpu);

        // ======================== CONCURRENT PROCESSING ========================

        // Arrays to store events for this batch
        cl_event *cpu_events = NULL;
        cl_event *gpu_events = NULL;
        
        if (num_images_cpu > 0) {
            cpu_events = (cl_event*)malloc(num_images_cpu * 3 * sizeof(cl_event));
        }
        if (num_images_gpu > 0) {
            gpu_events = (cl_event*)malloc(num_images_gpu * 3 * sizeof(cl_event));
        }

        int cpu_event_idx = 0;
        int gpu_event_idx = 0;

        // Launch ALL work asynchronously (non-blocking)
        for(int img_idx = 0; img_idx < batch_count; img_idx++) {
            
            // Pointer to this image in contiguous buffer
            unsigned char *in_ptr = batch_input + img_idx * image_size;
            unsigned char *out_ptr = batch_output + img_idx * image_size;
            
            // Decide which device processes this image
            int use_cpu = 0;

            if (mode == 1) {
                use_cpu = 1;
            } else if (mode == 2) {
                use_cpu = 0;
            } else {
                use_cpu = (img_idx < num_images_cpu);
            }
            
            if (use_cpu) {
                // ============ CPU processes this image ============
                
                err = clEnqueueWriteBuffer(q_cpu, buffer_cpu_input, CL_FALSE, 0,
                                          image_size, in_ptr,
                                          0, NULL, &cpu_events[cpu_event_idx++]);
                cl_error(err, "CPU write failed");
                
                err = clEnqueueNDRangeKernel(q_cpu, kernel_cpu, 2, NULL,
                                            global_size, local_size,
                                            0, NULL, &cpu_events[cpu_event_idx++]);
                cl_error(err, "CPU kernel launch failed");
                
                err = clEnqueueReadBuffer(q_cpu, buffer_cpu_output, CL_FALSE, 0,
                                          image_size, out_ptr,
                                          0, NULL, &cpu_events[cpu_event_idx++]);
                cl_error(err, "CPU read failed");
                
            } else {
                // ============ GPU processes this image ============
                
                err = clEnqueueWriteBuffer(q_gpu, buffer_gpu_input, CL_FALSE, 0,
                                          image_size, in_ptr,
                                          0, NULL, &gpu_events[gpu_event_idx++]);
                cl_error(err, "GPU write failed");
                
                err = clEnqueueNDRangeKernel(q_gpu, kernel_gpu, 2, NULL,
                                            global_size, local_size,
                                            0, NULL, &gpu_events[gpu_event_idx++]);
                cl_error(err, "GPU kernel launch failed");
                
                err = clEnqueueReadBuffer(q_gpu, buffer_gpu_output, CL_FALSE, 0,
                                          image_size, out_ptr,
                                          0, NULL, &gpu_events[gpu_event_idx++]);
                cl_error(err, "GPU read failed");
            }
        }

        // Wait for this batch to complete
        if (num_images_cpu > 0) clFinish(q_cpu);
        if (num_images_gpu > 0) clFinish(q_gpu);

        // ======================== COLLECT TIMING FROM EVENTS ========================

        // Process CPU events
        if (num_images_cpu > 0) {
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
        }

        // Process GPU events
        if (num_images_gpu > 0) {
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
        }

        // Cleanup events for this batch
        if (num_images_cpu > 0) {
            for(int i = 0; i < cpu_event_idx; i++) {
                clReleaseEvent(cpu_events[i]);
            }
            free(cpu_events);
        }
        if (num_images_gpu > 0) {
            for(int i = 0; i < gpu_event_idx; i++) {
                clReleaseEvent(gpu_events[i]);
            }
            free(gpu_events);
        }

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
    double time_cpu_total = 0;
    if (total_images_cpu > 0) {
        time_cpu_total = time_cpu_transfer_in + time_cpu_kernel + time_cpu_transfer_out;
        printf("2. CPU DEVICE (processed %d images)\n", total_images_cpu);
        printf("   Total CPU time:        %.2f ms\n", time_cpu_total);
        printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", 
              time_cpu_transfer_in, (time_cpu_transfer_in/time_cpu_total)*100);
        printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", 
              time_cpu_kernel, (time_cpu_kernel/time_cpu_total)*100);
        printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", 
              time_cpu_transfer_out, (time_cpu_transfer_out/time_cpu_total)*100);
        printf("   Average per image:     %.2f ms\n", time_cpu_total / total_images_cpu);
        printf("\n");
    }

    // GPU breakdown
    double time_gpu_total = 0;
    if (total_images_gpu > 0) {
        time_gpu_total = time_gpu_transfer_in + time_gpu_kernel + time_gpu_transfer_out;
        printf("3. GPU DEVICE (processed %d images)\n", total_images_gpu);
        printf("   Total GPU time:        %.2f ms\n", time_gpu_total);
        printf("   - Transfer IN:         %.2f ms (%.1f%%)\n", 
              time_gpu_transfer_in, (time_gpu_transfer_in/time_gpu_total)*100);
        printf("   - Kernel execution:    %.2f ms (%.1f%%)\n", 
              time_gpu_kernel, (time_gpu_kernel/time_gpu_total)*100);
        printf("   - Transfer OUT:        %.2f ms (%.1f%%)\n", 
              time_gpu_transfer_out, (time_gpu_transfer_out/time_gpu_total)*100);
        printf("   Average per image:     %.2f ms\n", time_gpu_total / total_images_gpu);
        printf("\n");
    }

    printf("====================\n");

    if (total_images_cpu > 0 && total_images_gpu > 0) {
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
    if (total_images_cpu > 0 && total_images_gpu > 0) {
        double t_cpu_per_image = time_cpu_total / total_images_cpu;
        double t_gpu_per_image = time_gpu_total / total_images_gpu;
        double optimal_gpu_ratio = t_cpu_per_image / (t_cpu_per_image + t_gpu_per_image);
        
        printf("8. OPTIMAL RATIO RECOMMENDATION\n");
        printf("   Based on measured performance:\n");
        printf("   CPU: %.3f ms/image\n", t_cpu_per_image);
        printf("   GPU: %.3f ms/image\n", t_gpu_per_image);
        printf("   Recommended GPU ratio: %.1f%%\n", optimal_gpu_ratio * 100);
        printf("   Run with: ./heterogeneous_blur both %.3f\n", optimal_gpu_ratio);
        printf("\n");
    }
    // ======================== CLEANUP ========================

    if (buffer_cpu_input)  clReleaseMemObject(buffer_cpu_input);
    if (buffer_cpu_output) clReleaseMemObject(buffer_cpu_output);
    if (buffer_gpu_input)  clReleaseMemObject(buffer_gpu_input);
    if (buffer_gpu_output) clReleaseMemObject(buffer_gpu_output);

    if (kernel_cpu) clReleaseKernel(kernel_cpu);
    if (kernel_gpu) clReleaseKernel(kernel_gpu);

    if (program_cpu) clReleaseProgram(program_cpu);
    if (program_gpu) clReleaseProgram(program_gpu);

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