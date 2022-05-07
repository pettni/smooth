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

#ifndef SMOOTH__SPLINE__DETAIL__CUMULATIVE_SPLINE_IMPL_HPP_
#define SMOOTH__SPLINE__DETAIL__CUMULATIVE_SPLINE_IMPL_HPP_

#include "../cumulative_spline.hpp"

namespace smooth {

template<std::size_t K, LieGroup G>
G cspline_eval_vs(
  std::ranges::sized_range auto && vs,
  const MatrixType auto & Bcum,
  Scalar<G> u,
  OptTangent<G> vel,
  OptTangent<G> acc) noexcept
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  assert(!acc.has_value() || vel.has_value());  // need vel for computation

  const auto U = monomial_derivatives<K, 2, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d2uvec(U[2].data());

  if (vel.has_value()) { vel.value().setZero(); }
  if (acc.has_value()) { acc.value().setZero(); }

  G g = Identity<G>(dof(*std::ranges::cbegin(vs)));

  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj = uvec.dot(Bcum.col(j));
    const G exp_Bt_v   = ::smooth::exp<G>(Bj * vj);
    g                  = composition(g, exp_Bt_v);
    if (vel.has_value()) {
      const Scalar<G> dBj = duvec.dot(Bcum.col(j));
      const auto Adj      = Ad(inverse(exp_Bt_v));
      vel.value().applyOnTheLeft(Adj);
      vel.value().noalias() += dBj * vj;
      if (acc.has_value()) {
        const Scalar<G> d2Bj = d2uvec.dot(Bcum.col(j));
        acc.value().applyOnTheLeft(Adj);
        acc.value().noalias() += dBj * ad<G>(vel.value()) * vj;
        acc.value().noalias() += d2Bj * vj;
      }
    }
  }

  return g;
}

template<std::size_t K, LieGroup G>
SplineJacobian<G, K - 1> cspline_eval_dg_dvs(
  std::ranges::sized_range auto && vs,
  const MatrixType auto & Bcum,
  const Scalar<G> & u,
  OptSplineJacobian<G, K - 1> dvel_dvs,
  OptSplineJacobian<G, K - 1> dacc_dvs) noexcept
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  assert(!dacc_dvs.has_value() || dvel_dvs.has_value());  // need vel for computation

  const auto U = monomial_derivatives<K, 2, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d2uvec(U[2].data());

  // derivatives w.r.t. vs
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * K> dg_dvs;
  dg_dvs.setZero();
  Eigen::Vector<Scalar<G>, Dof<G>> vel, acc;
  if (dvel_dvs.has_value()) {
    vel.setZero();
    dvel_dvs->setZero();
  }
  if (dacc_dvs.has_value()) {
    acc.setZero();
    dacc_dvs->setZero();
  }

  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj   = uvec.dot(Bcum.col(j));
    const Scalar<G> dBj  = duvec.dot(Bcum.col(j));
    const Scalar<G> d2Bj = d2uvec.dot(Bcum.col(j));

    const TangentMap<G> Adj   = Ad(::smooth::exp<G>(-Bj * vj));
    const TangentMap<G> DrExp = dr_exp<G>(-Bj * vj);

    dg_dvs.leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
    dg_dvs.template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * dr_exp<G>(Bj * vj);

    if (dvel_dvs.has_value() || dacc_dvs.has_value()) {
      dvel_dvs->leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
      dvel_dvs->template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * Adj * ad<G>(vel) * DrExp;
      dvel_dvs->template middleCols<Dof<G>>((j - 1) * Dof<G>) += dBj * TangentMap<G>::Identity();

      vel.applyOnTheLeft(Adj);
      vel += dBj * vj;

      if (dacc_dvs.has_value()) {
        dacc_dvs->leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
        dacc_dvs->leftCols(j * Dof<G>) -= dBj * ad<G>(vj) * dvel_dvs->leftCols(j * Dof<G>);
        dacc_dvs->template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * Adj * ad<G>(acc) * DrExp;
        dacc_dvs->template middleCols<Dof<G>>((j - 1) * Dof<G>) += dBj * ad<G>(vel);
        dacc_dvs->template middleCols<Dof<G>>((j - 1) * Dof<G>) += d2Bj * TangentMap<G>::Identity();

        acc.applyOnTheLeft(Adj);
        acc += dBj * ad<G>(vel) * vj + d2Bj * vj;
      }
    }
  }

  return dg_dvs;
}

template<std::size_t K, std::ranges::sized_range R, LieGroup G>
G cspline_eval_gs(
  R && gs, const MatrixType auto & Bcum, Scalar<G> u, OptTangent<G> vel, OptTangent<G> acc) noexcept
{
  assert(std::ranges::size(gs) == K + 1);

  static constexpr auto sub = [](const auto & x1, const auto & x2) { return rminus(x2, x1); };
  const auto vs             = gs | utils::views::pairwise_transform(sub);

  return composition(*std::ranges::begin(gs), cspline_eval_vs<K, G>(vs, Bcum, u, vel, acc));
}

template<std::size_t K, std::ranges::sized_range R, LieGroup G>
SplineJacobian<G, K> cspline_eval_dg_dgs(
  R && gs,
  const MatrixType auto & Bcum,
  const Scalar<G> & u,
  OptSplineJacobian<G, K> dvel_dgs,
  OptSplineJacobian<G, K> dacc_dgs) noexcept
{
  assert(std::ranges::size(gs) == K + 1);

  static constexpr auto sub = [](const auto & x1, const auto & x2) { return rminus(x2, x1); };
  const auto vs             = gs | utils::views::pairwise_transform(sub);

  SplineJacobian<G, K - 1> dvel_dvs;
  SplineJacobian<G, K - 1> dacc_dvs;
  auto dg_dvs = cspline_eval_dg_dvs<K, G>(vs, Bcum, u, dvel_dvs, dacc_dvs);

  // derivatives w.r.t. xs
  SplineJacobian<G, K> dg_dgs;
  dg_dgs.setZero();
  if (dvel_dgs.has_value()) { dvel_dgs->setZero(); }
  if (dacc_dgs.has_value()) { dacc_dgs->setZero(); }

  const auto U = monomial_derivatives<K, 0, Scalar<G>>(u);
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());

  G exp_series = Identity<G>();
  for (const auto & [j, vj] : utils::zip(std::views::iota(0u), vs)) {
    const TangentMap<G> DrExpinv = dr_expinv<G>(vj);
    const TangentMap<G> DlExpinv = dl_expinv<G>(vj);
    dg_dgs.template middleCols<Dof<G>>(j * Dof<G>) -=
      dg_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
    dg_dgs.template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
      dg_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DrExpinv;
    if (dvel_dgs.has_value()) {
      dvel_dgs->template middleCols<Dof<G>>(j * Dof<G>) -=
        dvel_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
      dvel_dgs->template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
        dvel_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DrExpinv;
    }
    if (dacc_dgs.has_value()) {
      dacc_dgs->template middleCols<Dof<G>>(j * Dof<G>) -=
        dacc_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
      dacc_dgs->template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
        dacc_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DrExpinv;
    }

    const Scalar<G> Bj = uvec.dot(Bcum.col(1 + j));
    exp_series         = composition(exp_series, smooth::exp<G>(Bj * vj));
  }

  // g also depends on g0 directly (in addition to through v1)
  dg_dgs.template leftCols<Dof<G>>() += Ad(inverse(exp_series));

  return dg_dgs;
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__DETAIL__CUMULATIVE_SPLINE_IMPL_HPP_
