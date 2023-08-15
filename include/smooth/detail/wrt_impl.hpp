// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <type_traits>

#include "../concepts/manifold.hpp"
#include "utils.hpp"

namespace smooth {
inline namespace v1_0 {

/**
 * @brief Compile-time size of a tuple of variables.
 *
 * If at least one variable is dynamically sized (Dof -1), this returns -1, otherwise returns sum of
 * all Dof's.
 */
template<typename Wrt>
constexpr int wrt_Dof()
{
  using DWrt = std::decay_t<Wrt>;

  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>) -> int {
    constexpr auto min_dof = std::min<int>({Dof<std::decay_t<std::tuple_element_t<Idx, DWrt>>>...});

    if constexpr (min_dof == -1) {
      return -1;
    } else {
      return (Dof<std::decay_t<std::tuple_element_t<Idx, DWrt>>> + ...);
    }
  };

  return f(std::make_index_sequence<std::tuple_size_v<DWrt>>{});
}

/**
 * @brief Cast a tuple of variables to a new scalar type.
 */
template<typename Scalar>
auto wrt_cast(auto && wrt)
{
  using Wrt = std::decay_t<decltype(wrt)>;

  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
    return std::make_tuple(cast<Scalar>(std::get<Idx>(wrt))...);
  };

  return f(std::make_index_sequence<std::tuple_size_v<Wrt>>{});
}

/**
 * @brief Calculate rplus(x_i, a[bi: bi + ni]) for a tuple
 */
template<typename Derived>
auto wrt_rplus(auto && wrt, const Eigen::MatrixBase<Derived> & a)
{
  using Wrt = std::decay_t<decltype(wrt)>;

  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
    static constexpr std::array<Eigen::Index, sizeof...(Idx)> Nx{Dof<std::decay_t<std::tuple_element_t<Idx, Wrt>>>...};

    const std::array<Eigen::Index, sizeof...(Idx)> ilen{dof(std::get<Idx>(wrt))...};
    const auto ibeg = utils::array_psum(ilen);

    // clang-format off
    return std::make_tuple(
      rplus(
        std::get<Idx>(wrt),
        a.template segment<std::get<Idx>(Nx)>(std::get<Idx>(ibeg), std::get<Idx>(ilen))
      )...
    );
    // clang-format on
  };

  return f(std::make_index_sequence<std::tuple_size_v<Wrt>>{});
}

// \cond
namespace detail {

template<typename T>
struct remove_const_ref
{
  using type = T;
};

template<typename T>
struct remove_const_ref<const T &>
{
  using type = T;
};

}  // namespace detail
// \endcond

/**
 * @brief Copy a tuple to make all elements modifiable.
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
constexpr auto wrt_copy_if_const(const std::tuple<T...> & in)
{
  return std::make_from_tuple<std::tuple<typename detail::remove_const_ref<T>::type...>>(in);
}

/**
 * @brief Copy a tuple to make all elements modifiable (rvalue version).
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
constexpr auto wrt_copy_if_const(std::tuple<T...> && in)
{
  return std::make_from_tuple<std::tuple<typename detail::remove_const_ref<T>::type...>>(std::move(in));
}

}  // namespace v1_0
}  // namespace smooth
