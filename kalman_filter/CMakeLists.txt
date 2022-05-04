cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(KalmanFilter CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # for find_package

# Set rpath to make things work without extra environment
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)
option(ENABLE_PYTHON OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr ")

include_directories(include)

add_subdirectory(start)
add_subdirectory(streams)
add_subdirectory(pinned_host_memory)

add_library(loader SHARED src/load.cpp)
target_link_libraries(loader PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS loader LIBRARY DESTINATION lib OPTIONAL)

if(ENABLE_PYTHON)
  # pybind11
  find_package(pybind11 REQUIRED)

  # numpy headers
  find_package(NumPy REQUIRED)

  include(pybind11Tools)
  pybind11_add_module(kalman_filter "src/bindings.cpp")
  target_link_libraries(kalman_filter PRIVATE loader)
  target_include_directories(kalman_filter PUBLIC ${PYBIND11_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
else()
  add_executable(KalmanFilter src/main.cpp)
  target_link_libraries(KalmanFilter PRIVATE loader)
  install(TARGETS KalmanFilter RUNTIME DESTINATION bin OPTIONAL)
endif()
