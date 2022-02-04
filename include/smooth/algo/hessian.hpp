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

template<LieGroup G>
inline auto algebra_generators = []() -> std::array<TangentMap<G>, Dof<G>> {
  std::array<TangentMap<G>, Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) { ret[i] = smooth::ad<G>(Tangent<G>::Unit(i)); }
  return ret;
}();

template<LieGroup G>
using hess_t = Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>>;

}  // namespace detail

/**
 * @brief Compute approximate Hessian (second derivative) of the exponential map.
 *
 * The computed Hessian is
 * \f[
 *   \mathrm{d}^{2r} \left( \exp a \right)_{aa}
 *   = \mathrm{d}^{2r} \left( x \oplus a \right)_{aa}
 * \f]
 */
template<LieGroup G, std::size_t Terms = std::tuple_size_v<decltype(detail::kBn)>>
  requires(Terms <= std::tuple_size_v<decltype(detail::kBn)>)
detail::hess_t<G> d2r_exp(const Tangent<G> & a)
{
  const TangentMap<G> ad_a_t = ad<G>(a).transpose();

  detail::hess_t<G> res;
  res.setZero();

  for (auto j = 0u; j < Dof<G>; ++j) {
    double coeff      = 1;                      // hold (-1)^i / i!
    Tangent<G> gi     = Tangent<G>::Unit(j);    // (ad_a^T)^i * e_j
    TangentMap<G> dgi = TangentMap<G>::Zero();  // right derivative of gi w.r.t. a

    for (auto iter = 0u; iter < Terms; ++iter) {
      res.template block<Dof<G>, Dof<G>>(0, j * Dof<G>) += coeff * dgi;
      dgi.applyOnTheLeft(ad_a_t);
      for (auto i = 0u; i < Dof<G>; ++i) {
        dgi.row(i).noalias() -= gi.transpose() * detail::algebra_generators<G>[i];
      }
      gi.applyOnTheLeft(ad_a_t);
      coeff *= (-1.) / (iter + 2);
    }
  }

  return res;
}

/**
 * @brief Compute derivative of the inverse of the derivative of the exponential map.
 *
 * The computed entity is
 * \f[
 *   \mathrm{d}^r \left( \mathrm{d}^r \exp_a^{-1} \right)_{a}
 * \f]
 */
template<LieGroup G, std::size_t Terms = std::tuple_size_v<decltype(detail::kBn)>>
  requires(Terms <= std::tuple_size_v<decltype(detail::kBn)>)
detail::hess_t<G> d2r_expinv(const Tangent<G> & a)
{
  const TangentMap<G> ad_a_t = ad<G>(a).transpose();

  detail::hess_t<G> res;
  res.setZero();

  for (auto j = 0u; j < Dof<G>; ++j) {
    double coeff      = 1;                      // hold (-1)^i / i!
    Tangent<G> gi     = Tangent<G>::Unit(j);    // (ad_a^T)^i * e_j
    TangentMap<G> dgi = TangentMap<G>::Zero();  // right derivative of gi w.r.t. a

    for (auto iter = 0u; iter < Terms; ++iter) {
      if (detail::kBn[iter] != 0) {
        res.template block<Dof<G>, Dof<G>>(0, j * Dof<G>) += (detail::kBn[iter] * coeff) * dgi;
      }
      dgi.applyOnTheLeft(ad_a_t);
      for (auto i = 0u; i < Dof<G>; ++i) {
        dgi.row(i).noalias() -= gi.transpose() * detail::algebra_generators<G>[i];
      }
      gi.applyOnTheLeft(ad_a_t);
      coeff *= (-1.) / (iter + 1);
    }
  }

  return res;
}

/**
 * @brief Compute approximate Hessian (second derivative) of rminus w.r.t. first argument.
 *
 * The computed Hessian is
 * \f[
 *   \mathrm{d}^{2r} \left( x \ominus_r y \right)_{xx}.
 * \f]
 */
template<LieGroup G, std::size_t Terms = std::tuple_size_v<decltype(detail::kBn)>>
  requires(Terms <= std::tuple_size_v<decltype(detail::kBn)>)
detail::hess_t<G> hessian_rminus(const G & x, const G & y)
{
  const Tangent<G> a           = rminus(x, y);
  const TangentMap<G> drexpinv = dr_expinv<G>(a);

  detail::hess_t<G> res = d2r_expinv<G>(a);
  for (auto j = 0u; j < Dof<G>; ++j) {
    res.template block<Dof<G>, Dof<G>>(0, j * Dof<G>).applyOnTheRight(drexpinv);
  }
  return res;
}

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
 * This function computes a finite number Terms of terms by recursively computing the inner
 * derivatives.
 */
template<LieGroup G, std::size_t Terms = std::tuple_size_v<decltype(detail::kBn)>>
  requires(Terms <= std::tuple_size_v<decltype(detail::kBn)>)
TangentMap<G> hessian_rminus_norm(const G & x, const G & y)
{
  const Tangent<G> a         = rminus(x, y);
  const TangentMap<G> ad_a_t = ad<G>(a).transpose();

  TangentMap<G> res = TangentMap<G>::Zero();
  double coeff      = 1;                          // hold (-1)^i / i!
  Tangent<G> gi     = a;                          // (ad_a^T)^i * a
  TangentMap<G> dgi = TangentMap<G>::Identity();  // right derivative of gi w.r.t. a

  for (auto iter = 0u; iter < Terms; ++iter) {
    if (detail::kBn[iter] != 0) { res += (detail::kBn[iter] * coeff) * dgi; }
    dgi.applyOnTheLeft(ad_a_t);
    for (auto i = 0u; i < Dof<G>; ++i) {
      dgi.row(i).noalias() -= gi.transpose() * detail::algebra_generators<G>[i];
    }
    gi.applyOnTheLeft(ad_a_t);
    coeff *= (-1.) / (iter + 1);
  }

  return res * dr_expinv<G>(a);
}

}  // namespace smooth

#endif  // SMOOTH__ALGO__HESSIAN_HPP_
