cmake_minimum_required(VERSION 3.22)
project(flowguard_opencl VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLOWGUARD_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

find_package(OpenCL REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
find_package(Threads REQUIRED)

add_library(flowguard_core
  src/scheduler.cpp
  src/risk_control.cpp
  src/protocol.cpp
  src/hardware_profile.cpp
  src/control_output.cpp
  src/artifacts.cpp
  src/opencl_pipeline.cpp
  src/dashboard.cpp
  src/web_server.cpp
)
target_include_directories(flowguard_core PUBLIC include)
target_link_libraries(flowguard_core PUBLIC OpenCL::OpenCL ${OpenCV_LIBS} Threads::Threads)
target_compile_definitions(flowguard_core PUBLIC
  CL_TARGET_OPENCL_VERSION=300
  BOOST_ERROR_CODE_HEADER_ONLY
  FLOWGUARD_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/perception.cl"
  FLOWGUARD_WEB_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/web/dist"
  FLOWGUARD_BLENDER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/simulator/flowguard_sim.py"
)
target_compile_options(flowguard_core PRIVATE -Wall -Wextra -Wpedantic)
if(FLOWGUARD_WARNINGS_AS_ERRORS)
  target_compile_options(flowguard_core PRIVATE -Werror)
endif()

add_executable(flowguard src/main.cpp src/application.cpp)
target_link_libraries(flowguard PRIVATE flowguard_core)
target_compile_options(flowguard PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  add_executable(flowguard_tests
    tests/test_main.cpp
    tests/test_scheduler.cpp
    tests/test_risk_control.cpp
    tests/test_protocol.cpp
    tests/test_hardware_profile.cpp
    tests/test_control_output.cpp
    tests/test_opencl.cpp
  )
  target_link_libraries(flowguard_tests PRIVATE flowguard_core)
  target_compile_options(flowguard_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME flowguard_unit COMMAND flowguard_tests)
  add_test(NAME flowguard_synthetic_replay
    COMMAND flowguard replay --synthetic expanding --frames 12 --mode cpu --no-dashboard)
  if(Python3_Interpreter_FOUND)
    add_test(NAME flowguard_px4_bridge_help
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integrations/px4_bridge.py --help)
    add_test(NAME flowguard_px4_bridge_unit
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_px4_bridge.py)
  endif()
endif()

install(TARGETS flowguard RUNTIME DESTINATION bin)
