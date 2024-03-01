cmake_minimum_required(VERSION 3.19)
project(eye_tracking)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV library
find_package( OpenCV 4 REQUIRED )
# Add header file
include_directories(include ${OpenCV_INCLUDE_DIRS} )

add_executable(eye_tracking main.cpp)
target_link_libraries(eye_tracking ${OpenCV_LIBS})