cmake_minimum_required (VERSION 3.0)
project (Filters)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set (Filters_VERSION_MAJOR 1)
set (Filters_VERSION_MINOR 0)

find_package(FFTW REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${FFTW_INCLUDE_DIRS})

add_executable(fft fft.cpp)

target_link_libraries(fft ${FFTW_LIBRARIES})
