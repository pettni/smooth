#ifndef SMOOTH__OPTIM__LM_HPP_
#define SMOOTH__OPTIM__LM_HPP_

#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include <iostream>
#include <numeric>

#include "smooth/concepts.hpp"
#include "smooth/diff.hpp"
#include "smooth/meta.hpp"

namespace smooth {

namespace detail {

/**
 * @brief Solve structured least-squares probelm
 *
 *   min_x \| [J ; diag(d)] x + [r; 0] \|^2
 *
 * Where a is a size N vector and J is M x N
 * and it must hold that J^T J + D^T D is positive semi-definite
 *
 * @param J_qr QR decomposition J P = Q R of J with column pivoting
 * @param d vector of length N representing diagonal matrix
 * @param r vector with same number of elements as there are rows in J
 * @param[out] Rt output matrix s.t. Rt' Rt = P' (J' J + D' D) P
 *
 * TODO change input args to take
 *  (
 *    Eigen::Matrix<double, N, N> & R,   // write resulting upper triangular matrix into here
 *    const Eigen::PermutationMatrix<N, N, PermIdx> & P
 *    const Eigen::Matrix<double, N, 1> & d,
 *    const Eigen::Matrix<double, N, 1> & Qt_r
 *  )
 */
template<int N, int M, typename QrType>
Eigen::Matrix<double, N, 1> solve_ls(const QrType & J_qr,
  const Eigen::Matrix<double, N, 1> & d,
  const Eigen::Matrix<double, M, 1> & r,
  std::optional<Eigen::Ref<Eigen::Matrix<double, N, N>>> Rt = {})
{
  // true if it's a sparse decomposition
  static constexpr bool is_sparse =
    std::is_base_of_v<Eigen::SparseMatrixBase<typename QrType::MatrixType>,
      typename QrType::MatrixType>;
  // type to use for A matrix
  using AType = std::conditional_t<is_sparse,
    Eigen::SparseMatrix<double, Eigen::RowMajor>,
    Eigen::Matrix<double, N, N>>;

  static constexpr int NM_min  = std::min(N, M);
  static constexpr int NM_rest = NM_min == -1 ? -1 : N - NM_min;

  // dynamic sizes
  const auto n       = J_qr.cols();
  const auto m       = J_qr.rows();
  const auto nm_min  = std::min(n, m);
  const auto nm_rest = n - nm_min;

  // form system
  // [A; B] x + [a; b]
  // where A = R
  //       B = P' diag(D) P = diag(P' D)
  //       a = Q' r
  //       b = 0
  AType A(n, n);
  if constexpr (is_sparse) {
    // allocate upper triangular pattern
    Eigen::Matrix<Eigen::Index, N, 1> pattern(n);
    for (auto i = 0u; i != n; ++i) { pattern(i) = i + 1; }
    A.reserve(pattern);
  } else {
    A.template bottomRows<NM_rest>(nm_rest).setZero();
  }
  A.template topRows<NM_min>(nm_min) = J_qr.matrixR().template topRows<NM_min>(nm_min);

  Eigen::Matrix<double, N, 1> a(n);
  a.template head<NM_min>(nm_min) = (J_qr.matrixQ().transpose() * r).template head<NM_min>(nm_min);
  a.template bottomRows<NM_rest>(nm_rest).setZero();

  // We operate on B and b row-wise, so just need to allocate one row
  Eigen::Matrix<double, N, 1> Bj(n);
  double bj;

  // QR decomposition of [A; B] with Givens rotations;
  // where it is known that A is upper triangular and B is diagonal
  // algorithm:
  //   R = A, Q = I
  //   for each nonzero in B part:
  //      find rotation G that eliminates nonzero without introducing new zeros
  //      Q = Q G
  //      R = G' R
  Eigen::JacobiRotation<double> G;
  for (auto j = 0u; j != n; ++j) {  // for each diagonal element
    // find permuted diagonal index
    const auto permidx = J_qr.colsPermutation().indices()(j);
    // initialize row j of B and b
    Bj.tail(n - 1 - j).setZero();
    Bj(j) = d(permidx);  // Bj(k) represents B(j, k)
    bj    = 0;           // bj represents b(j)

    for (auto col = j; col != n; ++col) {  // for each column right of diagonal
      if (A.coeff(col, col) >= 0 && Bj(col) == 0) { continue; }

      double r;
      G.makeGivens(A.coeff(col, col), Bj(col), &r);  // eliminates B(j, col)

      // perform matrix multiplication R = G' R
      // affects row 'col' of A and row 'j' of B
      A.coeffRef(col, col) = r;
      for (auto k = col + 1; k != n; ++k) {
        const double tmp   = G.c() * A.coeff(col, k) - G.s() * Bj(k);
        Bj(k)              = G.s() * A.coeff(col, k) + G.c() * Bj(k);
        A.coeffRef(col, k) = tmp;
      }

      // At the end we need Q matrix to multiply rhs as
      // Q' * rhs = (I * G0 * ... * Gk)' * rhs = Gk' * ... * G0' * rhs
      //
      // We can therefore do it as we go, here we set
      // rhs = G * rhs
      const double tmp = G.c() * a(col) - G.s() * bj;
      bj               = G.s() * a(col) + G.c() * bj;
      a(col)           = tmp;
    }
  }

  if constexpr (is_sparse) { A.makeCompressed(); }

  // solve triangular system A z = a to obtain z = R^-1 z
  // first check rank of upper-diagonal A (may happen if D not full-rank)
  int rank = 0;
  for (; rank != n && A.coeff(rank, rank) >= Eigen::NumTraits<double>::dummy_precision(); ++rank) {}

  Eigen::Matrix<double, N, 1> sol(n);
  sol.tail(n - rank).setZero();
  sol.head(rank) =
    A.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solve(a.head(rank));

  if (Rt.has_value()) { Rt.value() = A; }

  // solution is now equal to P z
  return -(J_qr.colsPermutation() * sol);
}

/**
 * @brief Approximate a Levenberg-Marquardt parameter lambda s.t. if x solves
 *
 *   \| [J; sqrt(lambda) * diag(D)] x  +  [r ; 0] \|^2
 *
 * then either
 *  * lambda = 0 AND \|diag(D) * x\| <= 0.1 Delta
 *    OR
 *  * lambda > 0 AND 0.9 Delta <= \|diag(D) * x\| <= 1.1 Delta
 *
 * @param J matrix MxN
 * @param d vector size Nx1
 * @param r vector size Mx1
 * @param Delta scalar
 * @return pair(lambda, x) where x is solves the least-squares problem for lambda
 *
 * TODO enable support for sparse J
 * */
template<int N, int M>
std::pair<double, Eigen::Matrix<double, N, 1>> lmpar(const Eigen::Matrix<double, M, N> J,
  const Eigen::Matrix<double, N, 1> & d,
  const Eigen::Matrix<double, M, 1> & r,
  double Delta)
{
  static constexpr int NM_min = std::min(N, M);

  // dynamic sizes
  const auto m      = J.rows();
  const auto n      = J.cols();
  const auto nm_min = std::min(n, m);

  // calculate qr decomposition of J
  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, M, N>> J_qr(J);

  Eigen::Matrix<double, NM_min, 1> Qt_r =
    (J_qr.matrixQ().transpose() * r).template head<NM_min>(nm_min);

  // calculate phi(0) by solving J x = -r as x = P R^-1 (-Q' r)
  Eigen::Matrix<double, N, 1> x(n);
  int rank     = J_qr.rank();
  x.head(rank) = J_qr.matrixR()
                   .topLeftCorner(rank, rank)
                   .template triangularView<Eigen::Upper>()
                   .solve(-Qt_r.head(rank));
  x.tail(n - rank).setZero();
  x.applyOnTheLeft(J_qr.colsPermutation());

  Eigen::Matrix<double, N, 1> D_x_iter = d.cwiseProduct(x);
  double D_x_iter_norm                 = D_x_iter.stableNorm();

  double alpha = 0;
  double phi   = D_x_iter_norm - Delta;
  double dphi;

  if (phi <= 0.1 * Delta) {
    // zero solution fulfills condition
    return std::make_pair(alpha, std::move(x));
  }

  // lower bound
  double l = 0;
  if (J_qr.rank() == n) {
    // full rank means we can calculate dphi(0)
    // as - \| Dx \| * \| Rinv (P' D' D x) / \| Dx \| \|^2
    Eigen::Matrix<double, N, 1> y =
      J_qr.colsPermutation().inverse() * (d.cwiseProduct(D_x_iter) / D_x_iter_norm);
    J_qr.matrixR()
      .template topLeftCorner<N, N>(n, n)
      .template triangularView<Eigen::Upper>()
      .transpose()
      .solveInPlace(y);
    dphi = -D_x_iter_norm * y.squaredNorm();
    l    = std::max(l, -phi / dphi);
  }

  // upper bound
  double u = (d.cwiseInverse().cwiseProduct(J.transpose() * r)).stableNorm() / Delta;

  // it typically converges in 2 or 3 iterations
  for (auto i = 0u; i != 20; ++i) {
    // ensure alpha stays within bounds (and not equal to zero)
    if (!(l < alpha && alpha < u)) { alpha = std::max<double>(0.001 * u, sqrt(l * u)); }

    // calculate phi
    Eigen::Matrix<double, N, N> Rt(n, n);

    x = solve_ls<N, M>(J_qr, sqrt(alpha) * d, r, Rt);

    D_x_iter      = d.cwiseProduct(x);
    D_x_iter_norm = D_x_iter.stableNorm();
    phi           = D_x_iter_norm - Delta;

    if (std::abs(phi) <= 0.1 * Delta) {
      break;  // condition fulfilled
    }

    // calculate derivative of phi
    Eigen::Matrix<double, N, 1> y =
      J_qr.colsPermutation().inverse() * (d.cwiseProduct(D_x_iter) / D_x_iter_norm);
    Rt.template triangularView<Eigen::Upper>().transpose().solveInPlace(y);
    dphi = -D_x_iter_norm * y.squaredNorm();

    // update bounds
    l = std::max<double>(l, alpha - phi / dphi);
    if (phi < 0) { u = alpha; }

    // update alpha
    alpha = alpha - ((phi + Delta) / Delta) * (phi / dphi);
  }

  return std::make_pair(alpha, std::move(x));
}

}  // namespace detail

/**
 * @brief Find a minimum of the non-linear least-squares problem
 *
 *  \min_{wrt...} \sum_i \| f(wrt...)_i \|^2
 *
 * @param f residuals to minimize
 * @param wrt arguments to f
 *
 * TODO split outer and inner loops, add printing, termination conditions, parameters
 */
template<typename _F, typename... _Wrt>
void minimize(_F && f, _Wrt &&... wrt)
{
  auto [r, J] = diff::dr(f, wrt...);

  double r_norm = r.stableNorm();

  // TODO how to initialize trust region?
  double Delta = 1;

  // scaling parameters
  auto diag = J.colwise().stableNorm().transpose().eval();

  for (int i = 0; i != 30; ++i) {
    const auto [lambda, a]  = detail::lmpar(J, diag, r, Delta);
    const double r_old_norm = r_norm;

    // take step a and re-evaluate function
    int segbeg        = 0;
    const auto f_iter = [&](auto && w) {
      static constexpr int Nx_j = detail::lie_info<std::decay_t<decltype(w)>>::lie_dof;
      const int nx_j            = detail::lie_info<std::decay_t<decltype(w)>>::lie_dof_dynamic(w);
      w += a.template segment<Nx_j>(segbeg, nx_j);
      segbeg += nx_j;
    };
    (f_iter(wrt), ...);
    std::tie(r, J) = diff::dr(f, wrt...);

    r_norm               = r.stableNorm();
    const double Da_norm = diag.cwiseProduct(a).stableNorm();

    // calculate actual to predicted reduction
    const double fra1 = Eigen::numext::abs2(r_norm / r_old_norm);
    const double fra2 = Eigen::numext::abs2((J * a).stableNorm() / r_old_norm);
    const double fra3 = Eigen::numext::abs2(std::sqrt(lambda) * Da_norm / r_old_norm);
    const double rho  = (1. - fra1) / (fra2 + 2. * fra3);

    // update trust region following Moré (1978)
    if (rho < 0.25) {
      double mu;
      if (r_norm <= r_old_norm) {
        mu = 0.5;
      } else if (r_norm <= 10 * r_old_norm) {
        const double gamma = -fra2 - fra3;
        mu                 = std::clamp(gamma / (2. * gamma + 1. - fra1), 0.1, 0.5);
      } else {
        mu = 0.1;
      }
      Delta *= mu;
    } else if ((lambda == 0 && rho < 0.75) || rho > 0.75) {
      Delta = 2 * Da_norm;
    }

    // update scaling
    diag = diag.cwiseMax(J.colwise().stableNorm().transpose());
  }
}

}  // namespace smooth

#endif  // SMOOTH__OPTIM__LM_HPP_
