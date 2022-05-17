// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <ranges>
#include <vector>

// convert a range into a std vector
template<std::ranges::range R>
auto r2v(R r)
{
  return std::vector(r.begin(), r.end());
}
