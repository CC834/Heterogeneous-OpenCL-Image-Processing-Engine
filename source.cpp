#define cimg_use_jpeg
#include <iostream>
#include "./CImg/CImg.h"
using namespace cimg_library;


int main(){
  CImg<unsigned char> img("test.png");  // Load image file "image.jpg" at object img

  std::cout << "Image width: " << img.width() << "Image height: " << img.height() << "Number of slices: " << img.depth() << "Number of channels: " << img.spectrum() << std::endl;  //dump some characteristics of the loaded image

  int i = 0;
  int j = 0;
  std::cout << std::hex << (int) img(i, j, 0, 0) << "\n" << std::endl;  //print pixel value for channel 0 (red) 
  std::cout << std::hex << (int) img(i, j, 0, 1) << "\n" << std::endl;  //print pixel value for channel 1 (green) 
  std::cout << std::hex << (int) img(i, j, 0, 2) << "\n" << std::endl;  //print pixel value for channel 2 (blue) 
  
  int h = 480;
  int w = 640;


  for(int i = 640/2-20; i < 640/2+20 ; i++) {
    img(i, 480/2, 0, 0) = 0;  // Red channel
    img(i, 480/2, 0, 1) = 0;  // Green channel
    img(i, 480/2, 0, 2) = 255;  // Blue channel
  }
  for(int j = h/2-20; j < h/2+20 ; j++) {
    img(w/2, j, 0, 0) = 0;  // Red channel
    img(w/2, j, 0, 1) = 0;  // Green channel
    img(w/2, j, 0, 2) = 255;  // Blue channel
  }

  img.display("My first CImg code");             // Display the image in a display window

  return 0;

}
