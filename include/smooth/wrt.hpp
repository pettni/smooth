// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__WRT_HPP_
#define SMOOTH__WRT_HPP_

#include <type_traits>

#include "internal/utils.hpp"
#include "manifold.hpp"

namespace smooth {

/**
 * @brief Grouping of function arguments.
 *
 * A tuple of references is created from the input arguments.
 */
template<typename... _Args>
  requires(Manifold<std::decay_t<_Args>> &&...)
auto wrt(_Args &&... args) { return std::forward_as_tuple(std::forward<_Args>(args)...); }

// \cond
namespace detail {

template<typename T>
struct wrt_dof_impl
{};

template<typename... Wrt>
struct wrt_dof_impl<std::tuple<Wrt...>>
{
  static constexpr int value =
    std::min<int>({Dof<std::decay_t<Wrt>>...}) == -1 ? -1 : (Dof<std::decay_t<Wrt>> + ...);
};

/**
 * @brief Trait for removing constness from reference types.
 */
template<typename T>
struct remove_const_ref
{
  using type = T;
};

/**
 * @brief Trait for removing constness from reference types.
 */
template<typename T>
struct remove_const_ref<const T &>
{
  using type = T;
};

}  // namespace detail
// \endcond

/**
 * @brief Compile-time size of a tuple of variables.
 *
 * If at least one variable is dynamically sized (Dof -1), this returns -1, otherwise returns sum of
 * all Dof's.
 */
template<typename Wrt>
constexpr int wrt_dof()
{
  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>)->int
  {
    constexpr auto min_dof =
      std::min<int>({Dof<std::decay_t<std::tuple_element_t<Idx, std::decay_t<Wrt>>>>...});

    if constexpr (min_dof == -1) {
      return -1;
    } else {
      return (Dof<std::decay_t<std::tuple_element_t<Idx, std::decay_t<Wrt>>>> + ...);
    }
  };

  return f(std::make_index_sequence<std::tuple_size_v<std::decay_t<Wrt>>>{});
}

/**
 * @brief Cast a tuple of variables to a new scalar type.
 */
template<typename Scalar, typename Wrt>
auto wrt_cast(Wrt && wrt)
{
  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>)
  {
    return std::make_tuple(cast<Scalar>(std::get<Idx>(wrt))...);
  };

  return f(std::make_index_sequence<std::tuple_size_v<std::decay_t<Wrt>>>{});
}

/**
 * @brief Calculate rplus(x_i, a[bi: bi + ni]) for a tuple
 */
template<typename Wrt, typename Derived>
auto wrt_rplus(Wrt && wrt, const Eigen::MatrixBase<Derived> & a)
{
  const auto f = [&]<std::size_t... Idx>(std::index_sequence<Idx...>)
  {
    static constexpr std::array<Eigen::Index, sizeof...(Idx)> Nx{
      Dof<std::decay_t<std::tuple_element_t<Idx, std::decay_t<Wrt>>>>...};

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

  return f(std::make_index_sequence<std::tuple_size_v<std::decay_t<Wrt>>>{});
}

/**
 * @brief Copy a tuple to make all elements modifiable.
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
std::tuple<typename detail::remove_const_ref<T>::type...> wrt_copy_if_const(
  const std::tuple<T...> & in)
{
  return std::make_from_tuple<std::tuple<typename detail::remove_const_ref<T>::type...>>(in);
}

/**
 * @brief Copy a tuple to make all elements modifiable (rvalue version).
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
std::tuple<typename detail::remove_const_ref<T>::type...> wrt_copy_if_const(std::tuple<T...> && in)
{
  return std::make_from_tuple<std::tuple<typename detail::remove_const_ref<T>::type...>>(
    std::move(in));
}

}  // namespace smooth

#endif  // SMOOTH__WRT_HPP_
