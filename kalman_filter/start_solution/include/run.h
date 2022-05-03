#pragma once

#include <string>

extern "C" int run(unsigned const max_events, std::string const input_path,
                   unsigned const n_repetitions, int const device_id,
                   int const n_streams);
