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

template<typename T>
struct wrt_dof
{};

/**
 * @brief Compile-time size of a tuple of variables.
 *
 * If at least one variable is dynamically sized (dof -1), this returns -1.
 */
template<typename... Wrt>
struct wrt_dof<std::tuple<Wrt...>>
{
  static constexpr int value =
    std::min<int>({Dof<std::decay_t<Wrt>>...}) == -1 ? -1 : (Dof<std::decay_t<Wrt>> + ...);
};

/**
 * @brief Cast a tuple of variables to a new scalar type.
 */
template<typename Scalar, typename... _Wrt>
auto wrt_cast(const std::tuple<_Wrt...> & wrt)
{
  std::tuple<CastT<Scalar, std::decay_t<_Wrt>>...> ret;
  utils::static_for<sizeof...(_Wrt)>(
    [&](auto i) { std::get<i>(ret) = cast<Scalar>(std::get<i>(wrt)); });
  return ret;
}

/**
 * @brief Add a tangent vector to a tuple of variables.
 */
template<typename Derived, typename... _Wrt>
auto wrt_rplus(const std::tuple<_Wrt...> & wrt, const Eigen::MatrixBase<Derived> & a)
{
  std::tuple<std::decay_t<_Wrt>...> ret;
  std::size_t i_beg = 0;
  utils::static_for<sizeof...(_Wrt)>([&](auto i) {
    using Wi             = std::decay_t<std::tuple_element_t<i, std::tuple<_Wrt...>>>;
    constexpr auto Ni    = Dof<Wi>;
    const std::size_t ni = dof(std::get<i>(wrt));
    std::get<i>(ret)     = rplus(std::get<i>(wrt), a.template segment<Ni>(i_beg, ni));
    i_beg += ni;
  });
  return ret;
}

/**
 * @brief Trait for removing constness from reference types.
 */
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

/**
 * @brief Copy a tuple to make all elements modifiable.
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
std::tuple<typename remove_const_ref<T>::type...> wrt_copy_if_const(const std::tuple<T...> & in)
{
  return std::make_from_tuple<std::tuple<typename remove_const_ref<T>::type...>>(in);
}

/**
 * @brief Copy a tuple to make all elements modifiable (rvalue version).
 *
 * Copies are created form const & members, rest is forwarded.
 */
template<typename... T>
std::tuple<typename remove_const_ref<T>::type...> wrt_copy_if_const(std::tuple<T...> && in)
{
  return std::make_from_tuple<std::tuple<typename remove_const_ref<T>::type...>>(std::move(in));
}

}  // namespace smooth

#endif  // SMOOTH__WRT_HPP_
