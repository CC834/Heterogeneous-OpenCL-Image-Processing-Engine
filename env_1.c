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

return 0;
}

