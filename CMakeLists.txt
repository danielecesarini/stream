cmake_minimum_required(VERSION 3.0.2)

project(stream
    VERSION 1.0
    DESCRIPTION "This is an OpenMP implementation of STREAM Benchmark"
    LANGUAGES C)

# Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(OpenMP REQUIRED C)
add_library(stream_openmp_c INTERFACE)
target_link_libraries(stream_openmp_c INTERFACE OpenMP::OpenMP_C)

# Source files
set(sources stream.c)

# Add executable
add_executable(stream ${sources})
target_link_libraries(stream PRIVATE stream_openmp_c)

# Install stream
install(
	TARGETS stream
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)