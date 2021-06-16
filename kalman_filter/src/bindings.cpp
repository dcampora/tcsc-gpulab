#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <numeric>
#include <cmath>

#include <load.h>

namespace {
  namespace py = pybind11;
  using std::vector;
  using std::array;
}


void run(std::string library,
         const unsigned max_events,
         std::string input_path,
         const unsigned  n_repetitions,
         const int device_id,
         const int n_streams)
{
  auto r = load(library);
  if (!r) {
    std::cout << "Failed to load library " << library << "\n";
    return;
  }

  if (input_path.back() != '/') {
    input_path += "/";
  }
  (*r)(max_events, input_path, n_repetitions, device_id, n_streams);
}

// Python Module and Docstrings
PYBIND11_MODULE(kalman_filter, m)
{
    m.doc() = R"pbdoc(
        Kalman Filter on GPUs exercise

        .. currentmodule:: kalman_filter

        .. autosummary::
           :toctree: _generate

           run
    )pbdoc";

    m.def("run", &run,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>{},
          "Run the Kalman Filter exercise\n"

          "Arguments:\n"
          "library name\n"
          "n_events\n"
          "data path\n"
          "n_repetitions\n"
          "device_id\n"
          "n_streams");
}
