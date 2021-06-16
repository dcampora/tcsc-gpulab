#include <string>
#include <iostream>
#include <load.h>

int main(int argc, char *argv[]) {

  if (argc != 6 &&  argc != 7) {
    std::cout << "Need five arguments: library number of events to process, input directory location, number of repititions and device to use" << std::endl;
    return -1;
  }

  const std::string library = argv[1];
  const unsigned max_events = atoi(argv[2]);
  std::string input_path = std::string(argv[3]);
  const unsigned n_repetitions = atoi(argv[4]);
  const int device_id = atoi(argv[5]);

  int n_streams = 0;
  if (argc == 7) {
    n_streams = atoi(argv[6]);
  }

  auto run = load(library);
  if (!run) {
    std::cout << "Failed to load library " << library << "\n";
  }

  if (input_path.back() != '/') {
    input_path += "/";
  }

  return (*run)(max_events, input_path, n_repetitions, device_id, n_streams);
}
