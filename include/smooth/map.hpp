// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

namespace smooth {

/**
 * @brief Memory mapping of internal Lie group types.
 */
template<typename T>
class Map;

/**
 * @brief Send Eigen types to Eigen::Map and other types to smooth::Map.
 */
template<typename T>
using MapDispatch = std::conditional_t<
  std::is_base_of_v<Eigen::MatrixBase<std::remove_const_t<T>>, std::remove_const_t<T>>,
  ::Eigen::Map<T>,
  ::smooth::Map<T>>;

}  // namespace smooth
