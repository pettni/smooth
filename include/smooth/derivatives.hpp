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

#ifndef SMOOTH__DERIVATIVES_HPP_
#define SMOOTH__DERIVATIVES_HPP_

#include <Eigen/Core>

#include "lie_group.hpp"

/**
 * @file
 * @brief Various useful derivatives.
 */

namespace smooth {

/**
 * @brief Derivative of matrix product.
 *
 * @param A matrix                                         [N x K]
 * @param dA derivative of A on horizontal Hessian form    [K x N*Nvar]
 * @param B matrix                                         [K x M]
 * @param dB derivative of B on horizontal Hessian form    [M x K*Nvar]
 * @return derivative of A * B on horizontal Hessian form
 */
template<typename At, typename dAt, typename Bt, typename dBt>
inline auto d_matrix_product(const At & A, const dAt & dA, const Bt & B, const dBt & dB)
{
  using Scalar = std::common_type_t<
    typename At::Scalar,
    typename dAt::Scalar,
    typename Bt::Scalar,
    typename dBt::Scalar>;

  static constexpr int N    = At::ColsAtCompileTime;
  static constexpr int M    = Bt::RowsAtCompileTime;
  static constexpr int Nvar = []() -> int {
    if constexpr (dAt::ColsAtCompileTime > 0 && N > 0) {
      return dAt::ColsAtCompileTime / N;
    } else if (dBt::ColsAtCompileTime > 0 && M > 0) {
      return dBt::ColsAtCompileTime / M;
    } else {
      return -1;
    }
  }();

  const auto n                  = A.cols();
  [[maybe_unused]] const auto k = A.rows();
  const auto m                  = B.rows();
  const auto nvar               = dA.cols() / (n);

  assert(k == B.cols());
  assert(nvar == dB.size() / (m * k));

  static constexpr int dAB_cols = (M > 0 && Nvar > 0) ? M * Nvar : -1;

  Eigen::Matrix<Scalar, N, dAB_cols> dAB = B.transpose() * dA;
  for (auto i = 0u; i < n; ++i) {
    for (auto j = 0u; j < m; ++j) {
      dAB.template middleCols<Nvar>(i * nvar, nvar) +=
        A(i, j) * dB.template middleCols<Nvar>(j * Nvar, nvar);
    }
  }
  return dAB;
}

/**
 * @brief Hessian of composed function \f$ (f \circ g)(x) \f$.
 *
 * @param Jf Jacobian of f at y = g(x)  [No x Ny   ]
 * @param Hf Hessian of f at y = g(x)   [Ny x No*Ny]
 * @param Jg Jacobian of g at x         [Ny x Nx   ]
 * @param Hg Hessian of g at x          [Nx x Ny*Nx]
 *
 * @return Hessian of size [No x No*Nx]
 */
template<typename JfT, typename HfT, typename JgT, typename HgT>
inline auto d2_fog(const JfT & Jf, const HfT & Hf, const JgT & Jg, const HgT & Hg)
{
  using Scalar = std::common_type_t<
    typename JfT::Scalar,
    typename HfT::Scalar,
    typename JgT::Scalar,
    typename HgT::Scalar>;

  static constexpr int No = JfT::RowsAtCompileTime;
  static constexpr int Ny = JfT::ColsAtCompileTime;
  static constexpr int Nx = JgT::ColsAtCompileTime;

  const auto no = Jf.rows();
  const auto ny = Jf.cols();

  [[maybe_unused]] const auto ni = Jg.rows();
  const auto nx                  = Jg.cols();

  // check some dimensions
  assert(ny == ni);
  assert(Hf.rows() == ny);
  assert(Hf.cols() == no * ny);
  assert(Hg.rows() == nx);
  assert(Hg.cols() == ni * nx);

  Eigen::Matrix<Scalar, Nx, (No == -1 || Nx == -1) ? -1 : No * Nx> ret(nx, no * nx);
  ret.setZero();

  for (auto i = 0u; i < no; ++i) {
    ret.template block<Nx, Nx>(0, i * nx, nx, nx) +=
      Jg.transpose() * Hf.template middleCols<Ny>(i * ny, ny) * Jg;
  }

  for (auto i = 0u; i < Jf.outerSize(); ++i) {
    for (Eigen::InnerIterator it(Jf, i); it; ++it) {
      ret.template block<Nx, Nx>(0, it.row() * nx) +=
        it.value() * Hg.template middleCols<Nx>(it.col() * nx, nx);
    }
  }

  return ret;
}

/**
 * @brief Jacobian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{r} (x \ominus_r y)_{x} \f$
 */
template<LieGroup G>
TangentMap<G> dr_rminus(const Tangent<G> & e)
{
  return dr_expinv<G>(e);
}
/**
 * @brief Hessian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} (x \ominus_r y)_{xx} \f$
 */
template<LieGroup G>
Hessian<G> d2r_rminus(const Tangent<G> & e)
{
  const auto J = dr_expinv<G>(e);

  auto res = d2r_expinv<G>(e);
  for (auto j = 0u; j < Dof<G>; ++j) {
    res.template block<Dof<G>, Dof<G>>(0, j * e.size(), e.size(), e.size()).applyOnTheRight(J);
  }
  return res;
}

/**
 * @brief Jacobian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^r \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_x \f$
 */
template<LieGroup G>
Eigen::RowVector<Scalar<G>, Dof<G>> dr_rminus_squarednorm(const Tangent<G> & e)
{
  return e.transpose() * dr_expinv<G>(e);
}

/**
 * @brief Hessian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_{xx} \f$
 */
template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>> d2r_rminus_squarednorm(const Tangent<G> & e)
{
  const TangentMap<G> J1 = dr_rminus<G>(e);   // N x N
  const Hessian<G> H1    = d2r_rminus<G>(e);  // N x (N*N)

  return d2_fog(e.transpose(), Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>>::Identity(), J1, H1);
}

}  // namespace smooth

#endif
