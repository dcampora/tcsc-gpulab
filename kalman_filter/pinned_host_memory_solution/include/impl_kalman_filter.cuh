#pragma once

#include <stdint.h>
#include <cmath>

#include "definitions.cuh"


__host__ __device__
void velo_kalman_filter_step(
  const float z,
  const float zhit,
  const float xhit,
  const float whit,
  float& x,
  float& tx,
  float& covXX,
  float& covXTx,
  float& covTxTx);

__host__ __device__
State simplified_fit(
  const Hit* host_hits,
  const MiniState& stateAtBeamLine,
  const Track& track);

__global__
void kalman_filter_gpu(
  const Hit* dev_hits,
  const Track* dev_tracks,
  const uint* event_offsets_hits,
  const uint* event_offsets_tracks,
  State* dev_states,
  const int max_events);
