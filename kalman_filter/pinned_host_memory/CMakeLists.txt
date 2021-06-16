include_directories(include)

set(kalman_filter_sources
  src/kalman_filter.cu
  src/utils.cu
  src/impl_kalman_filter.cu)

add_library(pinned_host_memory SHARED ${kalman_filter_sources})
install(TARGETS pinned_host_memory LIBRARY DESTINATION lib)
