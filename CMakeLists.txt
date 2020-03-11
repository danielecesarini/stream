cmake_minimum_required(VERSION 3.0.2)

project(stream
    VERSION 1.0
    DESCRIPTION "This is an OpenMP implementation of STREAM Benchmark"
    LANGUAGES C)

# Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Source files
set(sources stream.c)

# Add executable
add_executable(stream ${sources})

# MPI
find_package(MPI REQUIRED C)
target_link_libraries(stream PRIVATE MPI::MPI_C)

# OpenMP
if(STREAM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED C)
  target_link_libraries(stream PRIVATE OpenMP::OpenMP_C)
endif()

# Install stream
install(
	TARGETS stream
	RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)