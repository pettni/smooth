// Copyright (C) 2023 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Trust region algorithms for determining step size.
 */

#include <optional>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "smooth/version.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief Solve least-squares problem using LDL' decomposition.
 *
 * The problem is
 * @code
 *     [    J      ] dx  = [ -r ]                            (1)
 *     [ sqrt(λ) D ]       [  0 ]
 * @endcode
 *
 * @param[in] J matrix of size M x N
 * @param[in] d positive vector of size N representing diagonal of D
 * @param[in] r vector of size M
 * @param[in] lambda non-zero number
 * @param[out] dphi optionally calculate derivative ϕ'(λ)
 *
 * ϕ is the norm of the scaled solution as a function of λ:
 * @code
 *  ϕ(λ) = || D dx || = || D (J' J + λ D'D)^{-1} J' r ||
 * @endcode
 *
 * The system (1) is solved via LDLt factorization of the left-hand side of the
 * normal equations
 * @code
 *   (J' J + λ D' D) dx = -J' r.
 * @endcode
 */
template<typename D2, typename D3>
auto solve_linear_ldlt(
  const auto & J,
  const Eigen::MatrixBase<D2> & d,
  const Eigen::MatrixBase<D3> & r,
  const double lambda,
  std::optional<std::reference_wrapper<double>> dphi = {})
  -> Eigen::Vector<typename std::decay_t<decltype(J)>::Scalar, std::decay_t<decltype(J)>::ColsAtCompileTime>
{
  using JType             = std::decay_t<decltype(J)>;
  using Scalar            = typename JType::Scalar;
  static constexpr auto N = JType::ColsAtCompileTime;  // num variables

  static constexpr bool is_sparse = std::is_base_of_v<Eigen::SparseMatrixBase<JType>, JType>;

  using Ht = std::
    conditional_t<is_sparse, Eigen::SparseMatrix<typename JType::Scalar>, Eigen::Matrix<typename JType::Scalar, N, N>>;

  using LDLTt = std::conditional_t<is_sparse, Eigen::SimplicialLDLT<Ht>, Eigen::LDLT<Ht>>;

  Ht H = J.transpose() * J;
  for (auto i = 0u; i < H.rows(); ++i) { H.coeffRef(i, i) += lambda * d(i) * d(i); }

  const LDLTt ldlt(H);
  const Eigen::Vector<Scalar, N> x = ldlt.solve(-J.transpose() * r);

  if (dphi.has_value()) {
    const Eigen::Vector<Scalar, N> Dx  = -d.cwiseProduct(x);
    const Eigen::Vector<Scalar, N> d_q = d.cwiseProduct(Dx);
    const Eigen::Vector<Scalar, N> y   = ldlt.solve(d_q);
    dphi->get()                        = -d.cwiseProduct(Dx.normalized()).dot(y);
  }

  return x;
}

/**
 * @brief Approximately solve trust-region step determination problem.
 *
 * The step determination problem is to find the minimizing dx of
 *
 * @code
 *  min 0.5 || J dx  + r ||^2  s.t. || D dx || ≤ Δ          (1)
 * @endcode
 *
 * @param J matrix of size M x N
 * @param d vector of size N representing diagonal of D
 * @param r vector of size M
 * @param Delta trust region size
 *
 * @return {dx, λ} where dx is the (approximate) minimizer of (1) and λ is the corresponding
 * Lagrange multiplier for the inequality constraints.
 *
 * @note The current implementation sets the Lagrange multiplier to λ = 1 / Δ.
 *
 * # Theory
 *
 * It can be shown that (1) is equivalent to
 * @code
 *  min 0.5 ( || J dx + r ||^2 + λ || D dx ||^2 )           (2)
 * @endcode
 *
 * for some λ that satisfies the complimentarity condition
 *    λ (|| D dx || - Δ) = 0.
 *
 * in turn, (2) is equivalent to
 * @code
 *  min || [    J      ] dx  + [ r ] ||^2                   (3)
 *      || [ sqrt(λ) D ]       [ 0 ] ||
 * @endcode
 *
 * which is the least-squares solution to
 *
 * @code
 *  [    J      ] dx  = [ -r ]                              (4)
 *  [ sqrt(λ) D ]       [  0 ]
 * @endcode
 *
 * with normal equations
 *
 * @code
 *  (J' J + λ D' D) dx = - J' r                             (5)
 * @endcode
 */
template<typename D2, typename D3>
auto solve_trust_region(
  const auto & J, const Eigen::MatrixBase<D2> & d, const Eigen::MatrixBase<D3> & r, const double Delta) -> std::
  pair<Eigen::Vector<typename std::decay_t<decltype(J)>::Scalar, std::decay_t<decltype(J)>::ColsAtCompileTime>, double>
{
  const double lambda = 1. / Delta;

  const auto dx = solve_linear_ldlt(J, d, r, lambda);

  return {dx, lambda};
}

SMOOTH_END_NAMESPACE
