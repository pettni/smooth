// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "concepts/manifold.hpp"

namespace smooth {

/**
 * @brief Grouping of function arguments.
 *
 * A tuple of references is created from the input arguments.
 */
auto wrt(auto &&... args) requires(Manifold<std::decay_t<decltype(args)>> &&...)
{
  return std::forward_as_tuple(std::forward<decltype(args)>(args)...);
}

}  // namespace smooth
