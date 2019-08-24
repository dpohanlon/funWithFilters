cmake_minimum_required (VERSION 3.0)
project (Filters)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set (Filters_VERSION_MAJOR 1)
set (Filters_VERSION_MINOR 0)

find_package(FFTW REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${FFTW_INCLUDES})

add_executable(fft fft.cpp)

target_link_libraries(fft fftw3)
