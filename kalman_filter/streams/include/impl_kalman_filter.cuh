#pragma once

#include <stdint.h>
#include <cmath>

#include "definitions.cuh"


__host__ __device__
void velo_kalman_filter_step(
  const my_float_t z,
  const my_float_t zhit,
  const my_float_t xhit,
  const my_float_t whit,
  my_float_t& x,
  my_float_t& tx,
  my_float_t& covXX,
  my_float_t& covXTx,
  my_float_t& covTxTx);

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
  State* dev_states);
