#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>



// Scan all platforms and devices, print info, return first platform and device
int scan_platforms_and_devices(cl_platform_id *selected_platform, 
                                cl_device_id *selected_device) {
    cl_int err;
    
    // LOCAL variables - created inside function
    const cl_uint max_platforms = 10;
    cl_platform_id platforms[max_platforms];
    cl_uint n_platforms;
    
    const cl_uint max_devices = 10;
    cl_device_id devices[max_platforms][max_devices];
    cl_uint n_devices[max_platforms];
    
    // 1. Scan platforms
    err = clGetPlatformIDs(max_platforms, platforms, &n_platforms);
    if(err != CL_SUCCESS){
        printf("Error: Failed to scan for platforms\n");
        return -1;
    }
    
    printf("Number of available platforms: %d\n\n", n_platforms);
    
    // 2. For each platform, get info and scan devices
    for(int i = 0; i < n_platforms; i++){
        // Get platform name
        size_t name_size;
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &name_size);
        if(err != CL_SUCCESS) continue;
        
        char *platform_name = (char*)malloc(name_size);
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, platform_name, NULL);
        if(err == CL_SUCCESS){
            printf("\t[%d]-Platform Name: %s\n", i, platform_name);
        }
        free(platform_name);
        
        // Get devices for this platform
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, max_devices, 
                            devices[i], &n_devices[i]);
        if(err != CL_SUCCESS) continue;
        
        printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);
        
        // For each device, print info
        for(int j = 0; j < n_devices[i]; j++){
            size_t dev_name_size;
            err = clGetDeviceInfo(devices[i][j], CL_DEVICE_NAME, 0, NULL, &dev_name_size);
            if(err != CL_SUCCESS) continue;
            
            char *device_name = (char*)malloc(dev_name_size);
            err = clGetDeviceInfo(devices[i][j], CL_DEVICE_NAME, dev_name_size, device_name, NULL);
            if(err == CL_SUCCESS){
                printf("\t\t [%d]-Platform [%d]-Device: %s\n", i, j, device_name);
            }
            free(device_name);
        }
    }
    
    // 3. Return first platform and first device
    if(n_platforms > 0 && n_devices[0] > 0){
        *selected_platform = platforms[0];
        *selected_device = devices[0][0];
        return 0;
    }
    
    return -1;  // No platforms or devices found
}


int create_context_and_queue(cl_platform_id platform,
                             cl_device_id device,
                             cl_context *out_context,
                             cl_command_queue *out_queue)
{
    cl_int err;

    // 1. Create context
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };

    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    cl_error(err, "Failed to create a compute context\n");

    // 2. Create command queue
    cl_command_queue_properties proprt[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0
    };

    cl_command_queue queue =
        clCreateCommandQueueWithProperties(context, device, proprt, &err);
    cl_error(err, "Failed to create a command queue\n");

    *out_context = context;
    *out_queue   = queue;
    return 0;
}



// ====== LOAD KERNAL =====

int build_program_and_host_buffers(cl_context      context,
                                   cl_device_id    device,
                                   const char     *kernel_filename,
                                   unsigned int    count,
                                   cl_program     *out_program,
                                   float         **out_host_input,
                                   float         **out_host_output)
{
    cl_int err;

    // 1. Host buffers
    float *host_input  = (float*)malloc(count * sizeof(float));
    float *host_output = (float*)malloc(count * sizeof(float));
    if (!host_input || !host_output) {
        printf("Error: Failed to allocate host buffers\n");
        return -1;
    }

    for (unsigned int i = 0; i < count; i++) {
        host_input[i] = (float)i;
    }

    // 2. Load kernel source
    FILE *fileHandler = fopen(kernel_filename, "r");
    if (fileHandler == NULL) {
        printf("Error: Could not open %s\n", kernel_filename);
        free(host_input);
        free(host_output);
        return -1;
    }

    fseek(fileHandler, 0, SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    char *sourceCode = (char*)malloc(fileSize + 1);
    if (!sourceCode) {
        printf("Error: Failed to allocate source buffer\n");
        fclose(fileHandler);
        free(host_input);
        free(host_output);
        return -1;
    }

    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);

    // 3. Create program
    cl_program program =
        clCreateProgramWithSource(context, 1, (const char**)&sourceCode, NULL, &err);
    cl_error(err, "Failed to create program with source");
    free(sourceCode);

    // 4. Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *build_log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, build_log, NULL);
        build_log[log_size] = '\0';

        printf("Error: Failed to build program!\n");
        printf("Build Log:\n%s\n", build_log);
        free(build_log);
        // keep program so caller can release it if they want, or:
        clReleaseProgram(program);
        free(host_input);
        free(host_output);
        return -1;
    }

    printf("Program built successfully!\n");

    *out_program    = program;
    *out_host_input = host_input;
    *out_host_output= host_output;
    return 0;
}

//  ============== create kernal object ==============

int setup_kernel_and_buffers(cl_context        context,
                             cl_command_queue  command_queue,
                             cl_program        program,
                             const char       *kernel_name,
                             unsigned int      count,
                             float            *host_input,
                             cl_kernel        *out_kernel,
                             cl_mem           *out_device_input,
                             cl_mem           *out_device_output)
{
    cl_int err;

    // 1. Create kernel object
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    cl_error(err, "Failed to create kernel from the program");

    // 2. Create device buffers
    cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         sizeof(float) * count, NULL, &err);
    cl_error(err, "Failed to create input buffer on device");

    cl_mem device_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          sizeof(float) * count, NULL, &err);
    cl_error(err, "Failed to create output buffer on device");

    // 3. Copy host input to device input
    err = clEnqueueWriteBuffer(command_queue, device_input, CL_TRUE, 0,
                               sizeof(float) * count, host_input,
                               0, NULL, NULL);
    cl_error(err, "Failed to write input data to device");

    // 4. Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input);
    cl_error(err, "Failed to set kernel argument 0 (input)");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_output);
    cl_error(err, "Failed to set kernel argument 1 (output)");

    err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    cl_error(err, "Failed to set kernel argument 2 (count)");

    // 5. Output handles
    *out_kernel        = kernel;
    *out_device_input  = device_input;
    *out_device_output = device_output;

    return 0;
}


#endif // OPENCL_UTILS_H