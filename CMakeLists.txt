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
if(MPI_FOUND)
  target_link_libraries(stream 
    PRIVATE 
      ${MPI_C_LIBRARIES}
      ${MPI_C_LINK_FLAGS})
  target_include_directories(stream 
      PRIVATE
        ${MPI_C_INCLUDE_DIRS})
  target_compile_definitions(stream
    PRIVATE
      ${MPI_C_COMPILE_DEFINITIONS})
  target_compile_options(stream
    PRIVATE
      ${MPI_C_COMPILE_OPTIONS})
else()
  find_package(MPI REQUIRED C)
  target_link_libraries(stream PRIVATE MPI::MPI_C)
endif()

# OpenMP
if(STREAM_ENABLE_OPENMP)
  if(OpenMP_FOUND)
    target_link_libraries(stream 
      PRIVATE 
        ${OpenMP_C_LIBRARY}
        ${OpenMP_C_LIBRARIES}
        ${OpenMP_C_FLAGS})
    target_include_directories(stream
        PRIVATE
          ${OpenMP_C_INCLUDE_DIRS})
  else()
    find_package(OpenMP REQUIRED C)
    target_link_libraries(stream PRIVATE OpenMP::OpenMP_C)
  endif()
endif()

# Install stream
install(
	TARGETS stream
	RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)