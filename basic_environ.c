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
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

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
////////////////////////////////////////////////////////////////////////////////


/* ATTENTION: While prgramming in OpenCL it is a good idea to keep the reference manuals handy:
 * https://bashbaug.github.io/OpenCL-Docs/pdf/OpenCL_API.pdf
 * https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf (summary of OpenCL API)
 * https://www.khronos.org/assets/uploads/developers/presentations/opencl20-quick-reference-card.pdf
 */


int main(int argc, char** argv)
{
  int err;                              // error code returned from api calls
  size_t t_buf = 50;            // size of str_buffer
  char str_buffer[t_buf];       // auxiliary buffer 
  size_t e_buf;             // effective size of str_buffer in use
        
  size_t global_size;                       // global domain size for our calculation
  size_t local_size;                        // local domain size for our calculation

  const cl_uint num_platforms_ids = 10;             // max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];      // array of platforms
  cl_uint n_platforms;                      // effective number of platforms in use
  const cl_uint num_devices_ids = 10;               // max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids]; // array of devices
  cl_uint n_devices[num_platforms_ids];             // effective number of devices in use for each platform
    
  cl_device_id device_id;                           // compute device id 
  cl_context context;                               // compute context
  cl_command_queue command_queue;                   // compute command queue
    

  // 1. Scan the available platforms:
  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for (int i = 0; i < n_platforms; i++ ){
    //err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, /***???***/);
    err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
    cl_error (err, "Error: Failed to get info of the platform\n");
    printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
  }
  printf("\n");
  // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...
    
  // 2. Scan for devices in each platform
  for (int i = 0; i < n_platforms; i++ ){
    //err = clGetDeviceIDs( /***???***/, num_devices_ids, devices_ids[i], &(n_devices[i]));
    err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for(int j = 0; j < n_devices[i]; j++){
      // WHY TWO-CALL METHOD:
      // OpenCL requires the buffer size to exactly match or exceed the data size.
      // Problem with fixed buffers: sizeof(str_buffer) with variable-length arrays 
      // can be unreliable across compilers.
      // 
      // Solution - Two-call method:
      // 1st call: Ask OpenCL "how big is this string?" (pass size=0, buffer=NULL)
      // 2nd call: Allocate exact size, then retrieve the actual data
      // This guarantees the buffer size matches what OpenCL expects, avoiding CL_INVALID_VALUE errors.
      size_t device_name_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, 0, NULL, &device_name_size);
      cl_error(err, "clGetDeviceInfo: Getting device name size");
     

      // Allocate buffer of correct size
      char *device_name = (char*)malloc(device_name_size);

      // Second call: get actual name
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, device_name_size, device_name, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j, device_name);

      free(device_name);
    }
  } 
  // ***Task***: print on the screen the cache size, global mem size, local memsize, max work group size, profiling timer resolution and ... of each device

  // 3. Create a context, with a device
  //cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[/***???***/], 0};
  //context = clCreateContext(properties, /***???***/, NULL, NULL, &err);

  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
  context = clCreateContext(properties, n_devices[0], devices_ids[0], NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");

  // 4. Create a command queue
  cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  //command_queue = clCreateCommandQueueWithProperties( /***???***/, proprt, &err);
  command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
  cl_error(err, "Failed to create a command queue\n");

  // ============== LOAD TO KERNAL =============================

  /* It is still missing the runtime part of the OpenCL program: createBuffers, createProgram, createKernel, setKernelArg, ... */
  const unsigned int count = 1024;  // Number of elements to process
  
  // Create input and output arrays on the HOST (CPU)
  float *host_input = (float*)malloc(count * sizeof(float));   // Input data
  float *host_output = (float*)malloc(count * sizeof(float));  // Output results

  // Initialize input array with values (0, 1, 2, 3, ...)
  for(unsigned int i = 0; i < count; i++){host_input[i] = i;}

  // Load kernel source code from file
  FILE *fileHandler = fopen("kernel.cl", "r");
  if(fileHandler == NULL){ printf("Error: Could not open kernel.cl\n"); exit(-1); }

  // Get file size by seeking to end and checking position
  fseek(fileHandler, 0, SEEK_END);
  size_t fileSize = ftell(fileHandler);
  rewind(fileHandler);  // Go back to beginning

  // Read entire kernel source into a string buffer
  char *sourceCode = (char*)malloc(fileSize + 1);
  sourceCode[fileSize] = '\0';  // Null-terminate the string
  fread(sourceCode, sizeof(char), fileSize, fileHandler);
  fclose(fileHandler);

  // Create OpenCL program object from source code
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceCode, NULL, &err);
  cl_error(err, "Failed to create program with source");
  free(sourceCode);  // Don't need source string anymore


  // Step 3: Build (compile) the program for the device
  // This is like compiling C code - OpenCL compiles the kernel for the specific device
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  // Check if build failed and print the build log
  if (err != CL_SUCCESS){
    size_t log_size;
    // First get the size of the build log
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    // Allocate buffer for the log
    char *build_log = (char*)malloc(log_size + 1);
    
    // Get the actual build log
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';  // Null-terminate
    
    printf("Error: Failed to build program!\n");
    printf("Build Log:\n%s\n", build_log);
    free(build_log);
    exit(-1);
  }
  
  printf("Program built successfully!\n");

  // ========================Create The Kernal Object ========================

  // Create a kernel object from the compiled program
  // A "program" can contain multiple kernels, so we specify which one: "pow_of_two"
  cl_kernel kernel = clCreateKernel(program, "pow_of_two", &err);
  cl_error(err, "Failed to create kernel from the program");


  //Create memory buffers on the DEVICE (GPU/accelerator)
  // These are like mailboxes where the device can access data
  
  // Input buffer: Device will READ from this
  cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
  cl_error(err, "Failed to create input buffer on device");
  
  // Output buffer: Device will WRITE to this
  cl_mem device_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
  cl_error(err, "Failed to create output buffer on device");

  // Copy input data from host memory to device memory
  // This is like delivering the work materials to the worker's desk
  err = clEnqueueWriteBuffer(command_queue, device_input, CL_TRUE, 0, 
                             sizeof(float) * count, host_input, 0, NULL, NULL);
  cl_error(err, "Failed to write input data to device");

  // Set the arguments for the kernel function
  // The kernel function is: pow_of_two(float *in, float *out, unsigned int count)
  // We need to tell it where "in", "out", and "count" are
  
  // Argument 0: input buffer pointer
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input);
  cl_error(err, "Failed to set kernel argument 0 (input)");
  
  // Argument 1: output buffer pointer
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_output);
  cl_error(err, "Failed to set kernel argument 1 (output)");
  
  // Argument 2: count value
  err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
  cl_error(err, "Failed to set kernel argument 2 (count)");

  // =====================Launch the Kernel=======================
  
  //Launch the kernel on the device
  // We need to specify how many work-items (threads) to create
  
  local_size = 128;  // Work-items per work-group (like a team size)
  
  // Global size must be a multiple of local_size
  // Round up count to nearest multiple of local_size
  global_size = ((count + local_size - 1) / local_size) * local_size;
  
  printf("Launching kernel with global_size=%zu, local_size=%zu\n", global_size, local_size);
  
  // Launch the kernel!
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel");
  
  printf("Kernel launched successfully!\n");

  // ==============0 Read result from device to host ==============

  // Read the results back from device memory to host memory
  // This is like asking the worker to send back their completed work
  err = clEnqueueReadBuffer(command_queue, device_output, CL_TRUE, 0, 
                            sizeof(float) * count, host_output, 0, NULL, NULL);
  cl_error(err, "Failed to read output data from device");
  
  printf("Results copied back to host!\n");


  // Verify the results
  // Check if the first 10 results are correct
   printf("\nVerifying results (first 10 elements):\n");
   int correct = 1;  // Assume correct until proven wrong
   for(int i = 0; i < 10; i++){
     float expected = host_input[i] * host_input[i];  // What we expect
     float actual = host_output[i];                   // What we got

     printf("  input[%d] = %.0f, output[%d] = %.0f (expected %.0f) %s\n", 
            i, host_input[i], i, actual, expected, 
            (actual == expected) ? "OK" : "X");

     if(actual != expected){
       correct = 0;  // Found an error
     }
   }
   if(correct){
     printf("\nOK All results are correct!\n");
   } else {
     printf("\nX Some results are incorrect!\n");
   }

  // ================= Cleanup ======================
  
  // Cleanup : Realse all the allocated resources 
  // Step 11: Cleanup - release all allocated resources
  // Free device memory
  clReleaseMemObject(device_input);
  clReleaseMemObject(device_output);
  
  // Free OpenCL objects
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  
  // Free host memory
  free(host_input);
  free(host_output);
  
  printf("\nCleanup complete!\n");

  return 0;
}

