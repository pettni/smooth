// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "../cumulative_spline.hpp"

namespace smooth {
inline namespace v1_0 {

template<int K, LieGroup G>
  requires(K > 0)
G cspline_eval_vs(
  std::ranges::sized_range auto && vs,
  const MatrixType auto & Bcum,
  Scalar<G> u,
  OptTangent<G> vel,
  OptTangent<G> acc,
  OptTangent<G> jer) noexcept
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  assert(!acc.has_value() || vel.has_value());  // need vel for computation
  assert(!jer.has_value() || acc.has_value());  // need acc for computation

  const auto U = monomial_derivatives<K, 3, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d2uvec(U[2].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d3uvec(U[3].data());

  if (vel.has_value()) { vel.value().setZero(); }
  if (acc.has_value()) { acc.value().setZero(); }
  if (jer.has_value()) { jer.value().setZero(); }

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
        const Scalar<G> d2Bj            = d2uvec.dot(Bcum.col(j));
        const Tangent<G> vel_bracket_vj = ad<G>(vel.value()) * vj;
        acc.value().applyOnTheLeft(Adj);
        acc.value().noalias() += dBj * vel_bracket_vj;
        acc.value().noalias() += d2Bj * vj;
        if (jer.has_value()) {
          const Scalar<G> d3Bj = d3uvec.dot(Bcum.col(j));
          jer.value().applyOnTheLeft(Adj);
          jer.value().noalias() += 2 * dBj * ad<G>(acc.value()) * vj;
          jer.value().noalias() -= dBj * dBj * ad<G>(vel_bracket_vj) * vj;
          jer.value().noalias() += d2Bj * vel_bracket_vj;
          jer.value().noalias() += d3Bj * vj;
        }
      }
    }
  }

  return g;
}

template<int K, LieGroup G>
  requires(K > 0)
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

template<int K, std::ranges::sized_range R, LieGroup G>
  requires(K > 0)
G cspline_eval_gs(
  R && gs, const MatrixType auto & Bcum, Scalar<G> u, OptTangent<G> vel, OptTangent<G> acc, OptTangent<G> jer) noexcept
{
  assert(std::ranges::size(gs) == K + 1);

  static constexpr auto sub = [](const auto & x1, const auto & x2) { return rminus(x2, x1); };
  const auto vs             = gs | utils::views::pairwise_transform(sub);

  return composition(*std::ranges::begin(gs), cspline_eval_vs<K, G>(vs, Bcum, u, vel, acc, jer));
}

template<int K, std::ranges::sized_range R, LieGroup G>
  requires(K > 0)
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
  SplineJacobian<G, K - 1> dg_dvs = cspline_eval_dg_dvs<K, G>(vs, Bcum, u, dvel_dvs, dacc_dvs);

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
    const TangentMap<G> DlExpinv = -ad<G>(vj) + DrExpinv;  // cheaper formula
    dg_dgs.template middleCols<Dof<G>>(j * Dof<G>) -= dg_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
    dg_dgs.template middleCols<Dof<G>>((j + 1) * Dof<G>) += dg_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DrExpinv;
    if (dvel_dgs.has_value()) {
      dvel_dgs->template middleCols<Dof<G>>(j * Dof<G>) -= dvel_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
      dvel_dgs->template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
        dvel_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DrExpinv;
    }
    if (dacc_dgs.has_value()) {
      dacc_dgs->template middleCols<Dof<G>>(j * Dof<G>) -= dacc_dvs.template middleCols<Dof<G>>(j * Dof<G>) * DlExpinv;
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

}  // namespace v1_0
}  // namespace smooth
