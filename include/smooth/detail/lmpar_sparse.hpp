// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace smooth {
inline namespace v1_0 {
namespace detail {
/**
 * @brief Calculate the value and derivative of the function
 * \f[
 * \phi(\alpha) = \left\| D (J^T J + \alpha D^T D)^{-1} J^T r \right\| - \Delta
 * \f]
 *
 * @param J sparse matrix size MxN
 * @param d vector size N representing diagonal of D
 * @param r vector size M
 * @param Delta scalar
 * @param alpha scalar
 *
 * @return Triplet \f$(x, \phi(\alpha), \phi'(\alpha))\f$ where \f$x\f$ is a solution to \f$ J^T J +
 * \alpha D^T D = -J^T r\f$.
 */
template<int N, int M>
std::tuple<Eigen::Vector<double, N>, double, double> calc_phi(
  const auto & J, const Eigen::Vector<double, N> & d, const Eigen::Vector<double, M> & r, double Delta, double alpha)
{
  const auto n = J.cols();

  Eigen::SparseMatrix<double> lhs = J.transpose() * J;

  lhs.reserve(Eigen::Vector<double, N>::Ones(n));
  if (alpha > 0) {
    for (auto i = 0u; i != n; ++i) { lhs.coeffRef(i, i) += alpha * d(i) * d(i); }
  }
  lhs.makeCompressed();

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
  ldlt.compute(lhs);

  if (ldlt.info()) {
    // computation failed, add small diagonal to ensure positive definiteness
    for (auto i = 0u; i != n; ++i) { lhs.coeffRef(i, i) += Eigen::NumTraits<double>::epsilon(); }
    ldlt.compute(lhs);
  }

  // calculate q
  const Eigen::Vector<double, N> x = ldlt.solve(-J.transpose() * r);
  const Eigen::Vector<double, N> q = -d.cwiseProduct(x);

  // calculate phi
  const double phi = q.stableNorm() - Delta;

  // calculate dphi
  const Eigen::Vector<double, N> d_q = d.cwiseProduct(q);
  const Eigen::Vector<double, N> y   = ldlt.solve(d_q);
  const double dphi                  = -d.cwiseProduct(q.normalized()).dot(y);

  return std::make_tuple(x, phi, dphi);
}

/**
 * @brief Approximate a Levenberg-Marquardt parameter lambda s.t. if x solves
 *
 *   \| [J; sqrt(lambda) * diag(d)] x  +  [r ; 0] \|^2
 *
 * then either
 *  * lambda = 0 AND \|diag(d) * x\| <= 1.1 Delta
 *    OR
 *  * lambda > 0 AND 0.9 Delta <= \|diag(d) * x\| <= 1.1 Delta
 *
 * @param J sparse matrix MxN
 * @param d vector size Nx1
 * @param r vector size Mx1
 * @param Delta scalar
 *
 * @return pair(lambda, x) where x solves the least-squares problem for lambda
 */
template<int N, int M>
std::pair<double, Eigen::Vector<double, N>>
lmpar_sparse(const auto & J, const Eigen::Vector<double, N> & d, const Eigen::Matrix<double, M, 1> & r, double Delta)
{
  double alpha = 0;

  auto [x, phi, dphi] = calc_phi(J, d, r, Delta, alpha);

  if (phi <= 0.1 * Delta) {
    return std::make_pair(0, std::move(x));  // alpha = 0 solution fulfills condition
  }

  // initialize bounds
  double l = std::max<double>(0, -phi / dphi);
  double u = (d.cwiseInverse().cwiseProduct(J.transpose() * r)).stableNorm() / Delta;

  // it typically converges in 2 or 3 iterations
  for (auto i = 0u; i != 20; ++i) {
    // ensure alpha stays within bounds (and not equal to zero)
    if (!(l < alpha && alpha < u)) { alpha = std::max<double>(0.001 * u, sqrt(l * u)); }

    std::tie(x, phi, dphi) = calc_phi(J, d, r, Delta, alpha);

    if (std::abs(phi) <= 0.1 * Delta) {
      break;  // condition fulfilled
    }

    // update bounds
    l = std::max<double>(l, alpha - phi / dphi);
    if (phi < 0) { u = alpha; }

    // update alpha
    alpha = alpha - ((phi + Delta) / Delta) * (phi / dphi);
  }

  return std::make_pair(alpha, std::move(x));
}

}  // namespace detail
}  // namespace v1_0
}  // namespace smooth
