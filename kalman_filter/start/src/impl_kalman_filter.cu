/**
   Simple Kalman filter implementation for straight line tracks
   x and y are treated independently

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include "impl_kalman_filter.cuh"

/**
 * Helper function to filter one hit
 */
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
  my_float_t& covTxTx)
{
  // compute the prediction
  const my_float_t dz = zhit - z;
  const my_float_t predx = x + dz * tx;

  const my_float_t dz_t_covTxTx = dz * covTxTx;
  const my_float_t predcovXTx = covXTx + dz_t_covTxTx;
  const my_float_t dx_t_covXTx = dz * covXTx;

  const my_float_t predcovXX = covXX + 2 * dx_t_covXTx + dz * dz_t_covTxTx;
  const my_float_t predcovTxTx = covTxTx;
  // compute the gain matrix
  const my_float_t R = 1.0f / ((1.0f / whit) + predcovXX);
  const my_float_t Kx = predcovXX * R;
  const my_float_t KTx = predcovXTx * R;
  // update the state vector
  const my_float_t r = xhit - predx;
  x = predx + Kx * r;
  tx = tx + KTx * r;
  // update the covariance matrix. we can write it in many ways ...
  covXX /*= predcovXX  - Kx * predcovXX */ = (1 - Kx) * predcovXX;
  covXTx /*= predcovXTx - predcovXX * predcovXTx / R */ = (1 - Kx) * predcovXTx;
  covTxTx = predcovTxTx - KTx * predcovXTx;
}

/**
 * Fit the track with a Kalman filter,
 * allowing for some scattering at every hit
 */
__host__ __device__
State simplified_fit(
  const Hit* hits,
  const MiniState& stateAtBeamLine,
  const Track& track)
{
  const int nhits = track.hitsNum;
  const int first_hit = track.hits[0];
  const int last_hit = track.hits[nhits-1];

  /* Checker whether track is going backward or forward
     from the interaction point
  */

  const auto first_x = hits[first_hit].x;
  const auto first_y = hits[first_hit].y;
  const auto radius_first = std::sqrt(first_x*first_x + first_y*first_y);
  const auto last_x = hits[last_hit].x;
  const auto last_y = hits[last_hit].y;
  const auto radius_last = std::sqrt(last_x*last_x + last_y*last_y);

  const bool backward = radius_last < radius_first;
  const bool upstream = true;
  const int direction = (backward ? 1 : -1) * (upstream ? 1 : -1);
  const my_float_t noise2PerLayer =
    1e-8 + 7e-6 * (stateAtBeamLine.tx * stateAtBeamLine.tx + stateAtBeamLine.ty * stateAtBeamLine.ty);

  // assume the hits are sorted,
  // but want to get state closest to the beamline (upstream)
  // or farthest away from the beamline (downstream)
  // -> depending on whether the track goes forward or backward from the
  // interaction region
  // we have to swap the direction in which we loop over the hits
  // on the track
  int first_hit_index = 0;
  int last_hit_index = nhits - 1;
  int dhit = 1;
  if ((hits[ track.hits[last_hit_index] ].z - hits[ track.hits[first_hit_index] ].z) * direction < 0) {
    const int temp = first_hit_index;
    first_hit_index = last_hit_index;
    last_hit_index = temp;
    dhit = -1;
  }

  // We filter x and y simultaneously but take them uncorrelated.
  // filter first the first hit.
  State state;
  state.x = hits[ track.hits[first_hit_index] ].x;
  state.y = hits[ track.hits[first_hit_index] ].y;
  state.z = hits[ track.hits[first_hit_index] ].z;
  state.tx = stateAtBeamLine.tx;
  state.ty = stateAtBeamLine.ty;

  // Initialize the covariance matrix
  state.c00 = param_w_inverted;
  state.c11 = param_w_inverted;
  state.c20 = 0.f;
  state.c31 = 0.f;
  state.c22 = 1.f;
  state.c33 = 1.f;

  // add remaining hits
  for (uint i = first_hit_index + dhit; i != last_hit_index + dhit; i += dhit) {
    int hit_index = track.hits[i];
    const auto hit_x = hits[hit_index].x;
    const auto hit_y = hits[hit_index].y;
    const auto hit_z = hits[hit_index].z;

    // add the noise
    state.c22 += noise2PerLayer;
    state.c33 += noise2PerLayer;

    // filter X and filter Y
    velo_kalman_filter_step(
      state.z, hit_z, hit_x, param_w, state.x, state.tx, state.c00, state.c20, state.c22);
    velo_kalman_filter_step(
      state.z, hit_z, hit_y, param_w, state.y, state.ty, state.c11, state.c31, state.c33);

    // update z (note done in the filter, since needed only once)
    state.z = hit_z;
  }

  // add the noise at the last hit
  state.c22 += noise2PerLayer;
  state.c33 += noise2PerLayer;

  // finally, store the state
  return state;
}


__global__
void kalman_filter_gpu(
  const Hit* dev_hits,
  const Track* dev_tracks,
  const uint* event_offsets_hits,
  const uint* event_offsets_tracks,
  State* dev_states,
  const int max_events) {

  // Event loop
  for ( int event_number = 0; event_number < max_events; ++event_number ) {

    const Hit* hits_event = dev_hits + event_offsets_hits[event_number];
    const Track* tracks_event = dev_tracks + event_offsets_tracks[event_number];
    const int number_of_tracks = event_offsets_tracks[event_number+1] - event_offsets_tracks[event_number];

    // Every track will result in one state -> use same offsets for access
    State* states_event = dev_states + event_offsets_tracks[event_number];

// Track loop
    for ( int track_number = 0; track_number < number_of_tracks; ++track_number) {
      const MiniState state_at_beamline;
      const Track& track = tracks_event[track_number];

      State state = simplified_fit(hits_event, state_at_beamline, track);
      states_event[track_number] = state;
    }
  }
}
