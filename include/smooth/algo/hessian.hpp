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

#ifndef SMOOTH__ALGO__HESSIAN_HPP_
#define SMOOTH__ALGO__HESSIAN_HPP_

/**
 * @file
 * @brief Algorithms for computations of Hessians
 */

#include <smooth/lie_group.hpp>

namespace smooth {

namespace detail {

/// @brief The first Bernoulli numbers
static constexpr std::array<double, 23> kBn{
  1,               // 0
  -1. / 2,         // 1
  1. / 6,          // 2
  0.,              // 3
  -1. / 30,        // 4
  0,               // 5
  1. / 42,         // 6
  0,               // 7
  -1. / 30,        // 8
  0,               // 9
  5. / 66,         // 10
  0,               // 11
  -691. / 2730,    // 12
  0,               // 13
  7. / 6,          // 14
  0,               // 15
  -3617. / 510,    // 16
  0,               // 17
  43867. / 798,    // 18
  0,               // 19
  -174611. / 330,  // 20
  0,               // 21
  854513. / 138,   // 22
};

}  // namespace detail

/**
 * @brief Compute approximate Hessian (second derivative) of the squared norm of rminus.
 *
 * The computed Hessian is
 * \f[
 *   \mathrm{d}^{2r} \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_{xx}.
 * \f]
 *
 * @note The first derivative is
 * \f[
 *   \mathrm{d}^r \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_x = a^T \mathrm{d}^r \exp_a^{-1},
 * \f]
 * where \f$a = x \ominus_r y\f$, so the Hessian can be equivalently expressed as the the
 * first-order derivative \f[ \mathrm{d}^r \left( a^T \mathrm{d}^r \exp_a^{-1} \right)_x, \f]
 *
 * @tparam G Lie group
 * @tparam Terms number of terms in infinite sum to include (more terms -> better approximation)
 *
 * @param x first argument (derivative is w.r.t. x)
 * @param y second argument
 * @return The second derivative
 *
 * The Hessian can be expressed as the infinite sum
 * \f[
 *   \left( \sum_n B_n \frac{(-1)^n}{n!} \mathrm{d}^r \left( (ad_a^T)^n a \right)_a \right)
 * \mathrm{d}^r \exp_a^{-1}.
 * \f]
 * This function computes a finite number Terms of terms by recursively computing the inner derivatives.
 */
template<LieGroup G, std::size_t Terms = std::tuple_size_v<decltype(detail::kBn)>>
  requires(Terms <= std::tuple_size_v<decltype(detail::kBn)>)
TangentMap<G> hessian_rminus(const G & x, const G & y)
{
  using Tngt    = Tangent<G>;
  using TngtMap = TangentMap<G>;

  const Tngt a         = rminus(x, y);
  const TngtMap ad_a_t = ad<G>(a).transpose();

  const auto algebra_generators = []() -> std::array<TngtMap, Dof<G>> {
    std::array<TngtMap, Dof<G>> ret;
    for (auto i = 0u; i < Dof<G>; ++i) { ret[i] = smooth::ad<G>(Tngt::Unit(i)); }
    return ret;
  }();

  TngtMap res  = TngtMap::Zero();
  double coeff = 1;                    // hold (-1)^i / i!
  Tngt gi      = a;                    // (ad_a^T)^i * a
  TngtMap dgi  = TngtMap::Identity();  // right derivative of gi w.r.t. a

  for (auto iter = 0u; iter < Terms; ++iter) {
    if (detail::kBn[iter] != 0) { res += (detail::kBn[iter] * coeff) * dgi; }

    dgi.applyOnTheLeft(ad_a_t);

    for (auto i = 0u; i < smooth::Dof<G>; ++i) {
      dgi.row(i).noalias() -= gi.transpose() * algebra_generators[i];
    }

    gi.applyOnTheLeft(ad_a_t);

    coeff *= (-1.) / (iter + 1);
  }

  return res * dr_expinv<G>(a);
}

}  // namespace smooth

#endif  // SMOOTH__ALGO__HESSIAN_HPP_
