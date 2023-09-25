// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <type_traits>

#include <Eigen/Core>

namespace smooth {
inline namespace v1_0 {

namespace detail {

/**
 * @brief Trait class to mark external types as scalars.
 */
template<typename T>
struct scalar_trait
{
  /// @brief true if type T is a scalar
  static constexpr bool value = false;
};

}  // namespace detail

/**
 * @brief Concept to identify built-in scalars
 */
template<typename T>
concept ScalarType = std::is_floating_point_v<T> || detail::scalar_trait<T>::value;

/**
 * @brief Concept to identify Eigen matrices.
 */
template<typename G>
concept MatrixType = std::is_base_of_v<Eigen::MatrixBase<G>, G>;

/**
 * @brief Concept to identify Eigen column vectors.
 */
template<typename G>
concept RnType = MatrixType<G> && G::ColsAtCompileTime == 1;

}  // namespace v1_0
}  // namespace smooth
