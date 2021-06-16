/**
   Run Kalman filter on hits belonging to tracks of LHCb's Velo detector

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include "helpers.h"
#include "utils.h"
#include "impl_kalman_filter.cuh"
#include "kalman_filter.h"
#include <chrono>

using namespace std;

void kalman_filter_cpu(
  const Hit* host_hits,
  const Track* host_tracks,
  const std::vector<uint>& event_offsets_hits,
  const std::vector<uint>& event_offsets_tracks,
  State* states_cpu,
  const int max_events) {

  // Event loop
  for ( int event_number = 0; event_number < max_events; ++event_number ) {

    const Hit* host_hits_event = host_hits + event_offsets_hits[event_number];
    const Track* host_tracks_event = host_tracks + event_offsets_tracks[event_number];
    const int number_of_tracks = event_offsets_tracks[event_number+1] - event_offsets_tracks[event_number];

    // Every track will result in one state -> use same offsets for access
    State* states_event = states_cpu + event_offsets_tracks[event_number];

    // Track loop
    for ( int track_number = 0; track_number < number_of_tracks; ++track_number) {
      const MiniState state_at_beamline;
      const Track& track = host_tracks_event[track_number];

      State state = simplified_fit(host_hits_event, state_at_beamline, track);
      //std::cout << "at track " << track_number << ", tx = " << state.tx << ", ty = " << state.ty << ", x = " << state.x << ", y = " << state.y << ", z = " << state.z << std::endl;
      states_event[track_number] = state;
    }

  }
}


extern "C" int run(unsigned const max_events, std::string const input_path,
                   unsigned const n_repetitions, int const device_id,
                   int const)
{
  /* Chose device to use */
  CUDA_ASSERT( cudaSetDevice(device_id) );

  std::vector<std::string> folder_contents_hits = list_folder(input_path + "hits", "bin");
  std::vector<std::string> folder_contents_tracks = list_folder(input_path + "tracks", "bin");

  if ( max_events > folder_contents_hits.size() || max_events > folder_contents_tracks.size() ) {
    std::cout << "Requested " << max_events << " events, but only " << folder_contents_hits.size() << " are present in hits directory, and " << folder_contents_tracks.size() << " are present in tracks directory, stopping" << std::endl;
    return 1;
  }

  /* to do: allocate the host arrays that you think make sense to pinned memory*/
  Hit *host_hits = new Hit[max_hits_per_event * max_events]; // for all hits in all events
  Track *host_tracks = new Track[max_tracks_per_event * max_events]; // for all tracks in all events
  State *states_gpu = new State[max_tracks_per_event * max_events];
  State *states_cpu = new State[max_tracks_per_event * max_events];

  std::vector<uint> event_offsets_hits;
  get_hits(host_hits, folder_contents_hits, event_offsets_hits, input_path, max_events);

  std::vector<uint> event_offsets_tracks;
  get_tracks(host_tracks, folder_contents_tracks, event_offsets_tracks, input_path, max_events);

  /* Run Kalman filter on the CPU */
  kalman_filter_cpu(host_hits, host_tracks, event_offsets_hits, event_offsets_tracks, states_cpu, max_events);


  /* allocate device arrays */
  Hit *dev_hits;
  Track *dev_tracks;
  State *dev_states;
  uint* dev_event_offsets_hits;
  uint* dev_event_offsets_tracks;

  CUDA_ASSERT( cudaMalloc( (void**)&dev_hits, max_hits_per_event * max_events * sizeof(Hit) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&dev_tracks, max_tracks_per_event * max_events * sizeof(Track) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&dev_states, max_tracks_per_event * max_events * sizeof(State) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&dev_event_offsets_hits, (max_events + 1) * sizeof(uint) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&dev_event_offsets_tracks, (max_events + 1) * sizeof(uint) ) );

  const uint total_number_of_hits = event_offsets_hits.back();
  const uint total_number_of_tracks = event_offsets_tracks.back();

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for ( int repetition = 0; repetition < n_repetitions; ++repetition) {
    /* Copy hits, tracks and offsets from host to device */
    CUDA_ASSERT( cudaMemcpy( dev_hits, host_hits, total_number_of_hits * sizeof(Hit), cudaMemcpyHostToDevice ) );
    CUDA_ASSERT( cudaMemcpy( dev_tracks, host_tracks, total_number_of_tracks * sizeof(Track), cudaMemcpyHostToDevice ) );
    CUDA_ASSERT( cudaMemcpy( dev_event_offsets_hits, event_offsets_hits.data(), (max_events + 1) * sizeof(uint), cudaMemcpyHostToDevice ) );
    CUDA_ASSERT( cudaMemcpy( dev_event_offsets_tracks, event_offsets_tracks.data(), (max_events + 1) * sizeof(uint), cudaMemcpyHostToDevice ) );

    dim3 blocks(max_events);
    dim3 threads(32);
    kalman_filter_gpu<<<blocks, threads>>>(dev_hits, dev_tracks, dev_event_offsets_hits, dev_event_offsets_tracks, dev_states);

    /* Copy back states to host */
    CUDA_ASSERT( cudaMemcpy( states_gpu, dev_states, total_number_of_tracks * sizeof(State), cudaMemcpyDeviceToHost ) );

    cudaDeviceSynchronize();
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  /* Compare states computed on CPU and on GPU */
  compare_results(states_cpu, states_gpu, total_number_of_tracks);

  cout << "Total duration: " << elapsed_seconds.count() << " s " << endl;
  cout << "Time per event: " << elapsed_seconds.count() / max_events << endl;

  /* to do: Free host arrays correctly */
  delete [] host_hits;
  delete [] host_tracks;
  delete [] states_gpu;
  delete [] states_cpu;

  /* Free device arrays */
  CUDA_ASSERT( cudaFree( dev_hits ) );
  CUDA_ASSERT( cudaFree( dev_tracks ) );
  CUDA_ASSERT( cudaFree( dev_states ) );
  CUDA_ASSERT( cudaFree( dev_event_offsets_hits ) );
  CUDA_ASSERT( cudaFree( dev_event_offsets_tracks ) );

  return 0;
}
