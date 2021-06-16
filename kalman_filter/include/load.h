/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <functional>
#include <memory>

namespace {
  using run_function_t = std::function<int(
    unsigned const, std::string const,
    unsigned const, int const, int const)>;
}

using run_t = std::unique_ptr<run_function_t, std::function<void(run_function_t*)>>;
run_t load(std::string const& library);
