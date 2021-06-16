#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <dirent.h>
#include <regex>

#include "definitions.cuh"


std::vector<std::string> list_folder(const std::string& foldername, const std::string& extension);

void readFileIntoVector(const std::string& filename, std::vector<char>& events);

void get_hits(Hit *host_hits,
              HitsSoA &host_hits_soa,
              const std::vector<std::string> folderContents,
              std::vector<uint>& event_offsets_hits,
              const std::string input_path,
              const int max_events);

void get_tracks(Track *host_tracks,
              const std::vector<std::string> folderContents,
              std::vector<uint>& event_offsets_tracks,
              const std::string input_path,
              const int max_events);

void compare_results(State* cpu_states,
                     State* gpu_states,
                     const int total_number_of_states); 
