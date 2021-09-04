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

#ifndef SMOOTH__COMPAT__ODEINT_HPP_
#define SMOOTH__COMPAT__ODEINT_HPP_

/**
 * @file
 * @brief boost::odeint compatability header.
 */

#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

#include "smooth/manifold.hpp"

namespace smooth {

/**
 * @brief \p boost::odeint Stepper operations for Manifold types.
 *
 * \p boost::odeint Butcher tableaus are evaluated by weighted
 * calculations of the form y = Σ_{i=1}^n alpha_i x_i which are generically
 * implemented.
 *
 * However, for the special case of butcher tableaus it holds that alpha_1 = 1,
 * and furthermore x_1 is always of the state type while x_2 ... x_n are
 * of the derivative type. The scale sum can therefore be generalized to the Lie
 * group case as
 *
 *   y = x_1 * exp(Σ_{i=2}^n alpha_i x_i)
 *
 * The methods below inject those calculations into boost:odeint to enable
 * numerical integration on Lie groups. For succintness we implement a single
 * method using variadic templates.
 */
struct BoostOdeintOps
{
  /**
   * @brief Variadic scale_sum implementation.
   */
  template<typename... Fac>
  struct scale_sum
  {
    //! Storage for scale sum weights.
    const std::tuple<Fac...> m_alpha;

    //! Constructor for scale sum.
    inline scale_sum(Fac... alpha) noexcept : m_alpha(alpha...) {}

    //! Helper for scaled addition operation.
    template<typename... Ts, std::size_t... Is>
    inline auto helper(std::index_sequence<Is...>, const Ts &... as) noexcept
    {
      // plus 1 since alpha1 = 1 is not included in Ts...
      return ((std::get<Is + 1>(m_alpha) * as) + ...);
    }

    //! Scaled addition operation.
    template<Manifold T1, Manifold T2, typename... Ts>
      // \cond
      requires(
        std::is_same_v<T1, T2> && std::conjunction_v<std::is_same<typename T1::Tangent, Ts>...>)
    // \endcond
    inline void operator()(T1 & y, const T2 & x, const Ts &... as) noexcept
    {
      y = smooth::rplus(x, helper(std::make_index_sequence<sizeof...(Ts)>(), as...));
    }

    //! Required typedef.
    using result_type = void;
  };

  // \cond
  // clang-format off
  template<typename Fac1, typename Fac2 = Fac1>
  using scale_sum2 = scale_sum<Fac1, Fac2>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2>
  using scale_sum3 = scale_sum<Fac1, Fac2, Fac3>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3>
  using scale_sum4 = scale_sum<Fac1, Fac2, Fac3, Fac4>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4>
  using scale_sum5 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5>
  using scale_sum6 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6>
  using scale_sum7 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7>
  using scale_sum8 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8>
  using scale_sum9 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9>
  using scale_sum10 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10>
  using scale_sum11 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11>
  using scale_sum12 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11, typename Fac13 = Fac12>
  using scale_sum13 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12, Fac13>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11, typename Fac13 = Fac12, typename Fac14 = Fac13>
  using scale_sum14 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12, Fac13, Fac14>;
  // clang-format on
  // \endcond
};

}  // namespace smooth

/**
 * @brief SFINAE dispatcher for Manifold types.
 */
// \cond
template<smooth::Manifold G>
struct boost::numeric::odeint::operations_dispatcher_sfinae<G, void>
{
  using operations_type = ::smooth::BoostOdeintOps;
};
// \endcond

#endif  // SMOOTH__COMPAT__ODEINT_HPP_
