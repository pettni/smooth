#ifndef SMOOTH__INTERNAL__LMPAR_SPARSE_HPP_
#define SMOOTH__INTERNAL__LMPAR_SPARSE_HPP_

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace smooth::detail {

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
template<int N, int M, typename MatrixT>
std::tuple<Eigen::Matrix<double, N, 1>, double, double> calc_phi(const MatrixT & J,
  const Eigen::Matrix<double, N, 1> & d,
  const Eigen::Matrix<double, M, 1> & r,
  double Delta,
  double alpha)
{
  const auto n = J.cols();

  Eigen::SparseMatrix<double> lhs(n, n);
  lhs = J.transpose() * J;

  if (alpha > 0) {
    for (auto i = 0u; i != n; ++i) { lhs.coeffRef(i, i) += alpha * d(i) * d(i); }
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
  ldlt.compute(lhs);

  // calculate q
  Eigen::Matrix<double, N, 1> x = ldlt.solve(-J.transpose() * r);
  Eigen::Matrix<double, N, 1> q = -d.cwiseProduct(x);

  // calculate phi
  double q_norm = q.stableNorm();
  double phi    = q_norm - Delta;

  if (q_norm == 0) {
    return std::make_tuple(x, phi, 0);  // derivative doesn't matter here
  }

  // calculate dphi
  Eigen::Matrix<double, N, 1> d_q = d.cwiseProduct(q);
  Eigen::Matrix<double, N, 1> y   = ldlt.solve(d_q);

  Eigen::Matrix<double, 1, 1> dphi = -d_q.transpose() * y / q_norm;

  return std::make_tuple(x, phi, dphi(0));
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
template<int N, int M, typename MatrixT>
std::pair<double, Eigen::Matrix<double, N, 1>> lmpar_sparse(const MatrixT & J,
  const Eigen::Matrix<double, N, 1> & d,
  const Eigen::Matrix<double, M, 1> & r,
  double Delta)
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

}  // namespace smooth::detail

#endif  // SMOOTH__INTERNAL__LMPAR_SPARSE_HPP_
