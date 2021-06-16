/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <dlfcn.h>
#include <cstdio>
#include <iostream>

#include <load.h>

run_t load(std::string const& library)
{
  using run_function_t = run_t::element_type;
  run_t run {nullptr, [](run_function_t*) {}};

  Dl_info dl_info;
  dladdr((void*) load, &dl_info);

  std::string loader_path = dl_info.dli_fname;
  auto pos = loader_path.rfind("/");
  std::string lib_path = loader_path.substr(0, pos);
  const std::string library_path = lib_path + "/lib" + library + ".so";
    
  void* handle = dlopen(library_path.c_str(), RTLD_LAZY);

  if (!handle) {
    std::cout << "Cannot open library " << library_path << " : " << dlerror() << "\n";
    return run;
  }

  bool error = false;
  typedef int (*fun_t)(unsigned const, std::string const,
                       unsigned const, int const, int const);

  dlerror();
  fun_t run_function = reinterpret_cast<fun_t>(dlsym(handle, "run"));
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
    std::cout << "Cannot load symbol \"run\":" << dlsym_error << "\n";
    dlclose(handle);
  } else {
    run = run_t{new run_function_t{run_function},
              [handle](run_function_t* f) {
                delete f;
                dlclose(handle);
              }};
  }
  return run;
}
