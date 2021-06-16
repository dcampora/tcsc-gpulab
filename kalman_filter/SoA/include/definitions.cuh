#pragma once

#include <string>
#include <cassert>

typedef double my_float_t;

/* Cut-offs for array sizes */
static constexpr uint32_t max_tracks_per_event = 1200;
static constexpr uint32_t max_hits_per_event = 9500;
static constexpr uint32_t max_hits_per_track = 26;

/* Noise parameter for Kalman filter */
static constexpr my_float_t param_w = 3966.94f;
static constexpr my_float_t param_w_inverted = 0.000252083f;

/* Maximum allowed difference between results */
static constexpr my_float_t epsilon = 1.f;

struct State { 
  my_float_t x, y, z, tx, ty;
  my_float_t c00, c20, c22, c11, c31, c33;
};

struct MiniState {
  my_float_t x, y, z, tx, ty;

  __host__ __device__ 
MiniState() :
  x(0.f), y(0.f), z(0.f), tx(0.f), ty(0.f) {}
};

struct Hit {
  my_float_t x,y,z;
  unsigned int mcp;
};

/* Structure to hold the pointers to the arrays */
struct HitsSoA {
  my_float_t *x, *y, *z;
  unsigned int *mcp;
};

struct HitZAndIndex {
  my_float_t z;
  int index;

  __host__ __device__ 
HitZAndIndex(const my_float_t z_, const int index_) :
  z(z_), index(index_) {};
};

struct Track{
  int hitsNum = 0;
  int hits[max_hits_per_track];
  unsigned int mcp;

  __host__ __device__
  void add_hit(const int hit_index) {
    if ( hitsNum >= max_hits_per_track )
      printf("hitsNum = %u, max hits per track = %u \n", hitsNum, max_hits_per_track);
    assert( hitsNum < max_hits_per_track);
    hits[hitsNum++] = hit_index;
}

};
