include_directories(include)

set(kalman_filter_sources
  src/kalman_filter.cu
  src/utils.cu
  src/impl_kalman_filter.cu)

add_library(start SHARED ${kalman_filter_sources})
install(TARGETS start LIBRARY DESTINATION lib)
