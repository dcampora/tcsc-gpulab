#pragma once

#include <string>

#include "utils.h"
#include "definitions.cuh"
#include "impl_kalman_filter.cuh"

void kalman_filter_cpu(
  const Hit* host_hits,
  const Track* host_tracks,
  const std::vector<uint>& event_offsets_hits,
  const std::vector<uint>& event_offsets_tracks,
  State* states_cpu,
  const int max_events);
