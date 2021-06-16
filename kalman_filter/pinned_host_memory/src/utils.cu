/**
   Utils for reading in data from binary files

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include "utils.h"

std::vector<std::string> list_folder(const std::string& foldername, const std::string& extension)
{
  std::vector<std::string> folderContents;
  DIR* dir;
  struct dirent* ent;
  std::string suffix = std::string {"."} + extension;
  // Find out folder contents
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = ent->d_name;
      if (filename != "." && filename != "..") {
        folderContents.emplace_back(filename);
      }
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      std::cout << "No " << extension << " files found in folder " << foldername << std::endl;
      return folderContents;
    }
    else {
      std::cout << "Found " << folderContents.size() << " files" << std::endl;
    }
  }
  else {
    std::cout << "Folder " << foldername << " could not be opened" << std::endl;
    return folderContents;
  }

  // Sort files by natural order

  const std::regex bin_format {"(\\d+)(?:_(\\d+))?\\.bin"};

  // Convert "N.bin" to (0, N) and "N_M.bin" to (N, M)
  auto name_to_number = [bin_format](const std::string& arg) -> std::pair<int, long long> {
    std::smatch m;
    if (!std::regex_match(arg, m, bin_format)) {
      return {0, 0};
    }
    else if (m.length(2) == 0) {
      return {0, std::stol(std::string {m[1].first, m[1].second})};
    }
    else {
      return {std::stoi(std::string {m[1].first, m[1].second}), std::stol(std::string {m[2].first, m[2].second})};
    }
  };

  // Sort in natural order by converting the filename to a pair of (int, long long)
  auto natural_order = [name_to_number](const std::string& lhs, const std::string& rhs) -> bool {
    return std::less<std::pair<int, long long>> {}(name_to_number(lhs), name_to_number(rhs));
  };

  // Sort folder contents (file names)
  std::sort(folderContents.begin(), folderContents.end(), natural_order);

  return folderContents;
}

void readFileIntoVector(const std::string& filename, std::vector<char>& events)
{
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  events.resize(dataSize);
  infile.read((char*) &(events[0]), dataSize);
  infile.close();
}

void get_hits(Hit *host_hits,
              const std::vector<std::string> folderContents,
              std::vector<uint>& event_offsets_hits,
              const std::string input_path,
              const int max_events) {

  // Loop over events
  unsigned int accumulated_size = 0;
  for ( int event_number = 0; event_number < max_events; ++event_number ) {

    std::string filename = input_path + "hits/" + folderContents[event_number];
    int accumulated_hit_number = 0;

    std::vector<char> event;
    readFileIntoVector(filename, event);
    uint8_t* input = (uint8_t*) event.data();

    int n_hits = *((uint32_t*) input);
    input += sizeof(uint32_t);

    for ( uint32_t i = 0; i < n_hits; ++i ) {
      Hit hit;
      hit.x = *((float*) input);
      input += sizeof(float);
      hit.y = *((float*) input);
      input += sizeof(float);
      hit.z = *((float*) input);
      input += sizeof(float);
      hit.mcp = *((uint32_t*) input);
      input += sizeof(uint32_t);

      host_hits[accumulated_size + accumulated_hit_number] = hit;

      ++accumulated_hit_number;
      if ( accumulated_hit_number >= max_hits_per_event ) break;
    }

    // Save offset to where the hits start for every event
    const unsigned int event_size = accumulated_hit_number;
    event_offsets_hits.push_back(accumulated_size);
    accumulated_size += event_size;
  }

  // Add last offset
  event_offsets_hits.push_back(accumulated_size);

}

void get_tracks(Track *host_tracks,
              const std::vector<std::string> folderContents,
              std::vector<uint>& event_offsets_tracks,
              const std::string input_path,
              const int max_events) {

  // Loop over events
  unsigned int accumulated_size = 0;
  for ( int event_number = 0; event_number < max_events; ++event_number ) {

    std::string filename = input_path + "tracks/" + folderContents[event_number];
    int accumulated_track_number = 0;

    std::vector<char> event;
    readFileIntoVector(filename, event);
    uint8_t* input = (uint8_t*) event.data();

    uint32_t n_tracks = *((uint32_t*) input);
    input += sizeof(uint32_t);

    for ( uint32_t i = 0; i < n_tracks; ++i ) {
      Track track;
      track.hitsNum = *((int*) input);
      input += sizeof(int);
      for ( int j = 0; j < track.hitsNum; ++j ) {
        track.hits[j] = *((int*) input);
        input += sizeof(int);
        assert(track.hits[j] < max_hits_per_event);
      }
      track.mcp = *((uint32_t*) input);
      input += sizeof(uint32_t);

      host_tracks[accumulated_size + accumulated_track_number] = track;

      ++accumulated_track_number;
      if ( accumulated_track_number >= max_tracks_per_event ) break;
    }

    // Save offset to where the hits start for every event
    const unsigned int event_size = accumulated_track_number;
    event_offsets_tracks.push_back(accumulated_size);
    accumulated_size += event_size;
  }

  // Add last offset
  event_offsets_tracks.push_back(accumulated_size);

}

void compare_results(State* cpu_states,
                     State* gpu_states,
                     const int total_number_of_states) {
  for ( int i = 0; i < total_number_of_states; ++i ) {
    const State& cpu_state = cpu_states[i];
    const State& gpu_state = gpu_states[i];
    if ( std::abs(cpu_state.x - gpu_state.x) > epsilon) {
      std::cout << "Difference in x: " << cpu_state.x << " vs " << gpu_state.x << std::endl;
    }
    if ( std::abs(cpu_state.y - gpu_state.y) > epsilon) {
      std::cout << "Difference in y: " << cpu_state.y << " vs " << gpu_state.y << std::endl;
    }
    if ( std::abs(cpu_state.z - gpu_state.z) > epsilon) {
      std::cout << "Difference in z: " << cpu_state.z << " vs " << gpu_state.z << std::endl;
    }
    if ( std::abs(cpu_state.tx - gpu_state.tx) > epsilon) {
      std::cout << "Difference in tx: " << cpu_state.tx << " vs " << gpu_state.tx << std::endl;
    }
    if ( std::abs(cpu_state.ty - gpu_state.ty) > epsilon) {
      std::cout << "Difference in ty: " << cpu_state.ty << " vs " << gpu_state.ty << std::endl;
    }

  }

}
