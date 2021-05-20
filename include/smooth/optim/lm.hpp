#ifndef SMOOTH__OPTIM__LM_HPP_
#define SMOOTH__OPTIM__LM_HPP_

#include <Eigen/Core>
#include <Eigen/Jacobi>

#include <iostream>


/**
 * @brief Solve structured least-squares probelm
 *
 *   min_a \| [J ; diag(d)] a + [r; 0] \|^2
 *
 * Where a is a size N vector
 *
 * @param J_qrdecomp QR decomposition of J with column pivoting
 * @param d vector of length N representing diagonal matrix
 * @param r vector with same number of elements as there are rows in J
 *
 * Currenly only supports square J matrices
 */
template<int N>
Eigen::Matrix<double, N, 1> solve_ls(
  const Eigen::ColPivHouseholderQR<Eigen::Matrix<double, N, N>> & J_qrdecomp,
  const Eigen::Matrix<double, N, 1> & d,
  const Eigen::Matrix<double, N, 1> & r
) {
  Eigen::Matrix<double, N, N> upper = J_qrdecomp.matrixR();
  Eigen::Matrix<double, N, N> lower = (J_qrdecomp.colsPermutation().transpose() * d).asDiagonal();

  Eigen::Matrix<double, N, 1> rhs_upper, rhs_lower;
  rhs_upper = J_qrdecomp.matrixQ().transpose() * r;
  rhs_lower.setZero();

  Eigen::JacobiRotation<double> G;

  // QR decomposition of A with Givens rotations;
  // algorithm:
  //   R = A, Q = I
  //   for each nonzero in lower part:
  //      find rotation G that eliminates nonzero without introducing new zeros
  //      Q = Q G
  //      R = G' R
  for (auto col = 0u; col != N; ++col)  // for each column
  {
    for (auto row = 0u; row != col + 1; ++row) {  // for each row above diagonal
      G.makeGivens(upper(col, col), lower(row, col));  // eliminates lower(row, col)

      // perform matrix multiplication R = G' R
      upper(col, col) = G.c() * upper(col, col) - G.s() * lower(row, col);
      for (int i = col + 1; i < N; ++i) {
        double tmp = G.c() * upper(col, i) - G.s() * lower(row, i);
        lower(row, i) = G.s() * upper(col, i) + G.c() * lower(row, i);
        upper(col, i) = tmp;
      }

      // At the end we need Q matrix to multiply rhs as
      // Q' * rhs = (I * Q0 * ... * Qk)' * rhs = Qk' * ... * Q0' * rhs
      //
      // We can therefore do it as we go, here we set
      // rhs = Q' * rhs
      double tmp = G.s() * rhs_upper(col) + G.c() * rhs_lower(row);
      rhs_upper(col) = G.c() * rhs_upper(col) - G.s() * rhs_lower(row);
      rhs_lower(row) = tmp;
    }
  }

  // solve triangular system R z = rhs to obtain z = R^-1 z
  upper.template triangularView<Eigen::Upper>().solveInPlace(rhs_upper);

  // solution is now equal to P z
  return -(J_qrdecomp.colsPermutation() * rhs_upper);
}


template<typename _F>
void optimize(_F && f)
{
  using F = std::decay_t<_F>;
  using G = typename F::Group;
  using Tangent = typename G::Tangent;

  using R = decltype(f(G{}));

  static constexpr int num_res = R::SizeAtCompileTime;

  using Jac = Eigen::Matrix<double, num_res, G::lie_dof>;

  // optimization variable
  G g = G::Random();
  Jac J = f.df(g);
  R r = f(g);

  // trust region
  double Delta = 1;

  // scaling parameters
  Eigen::Matrix<double, G::lie_dof, 1> diag = J.colwise().stableNorm();

  std::cout << "starting parameters" << std::endl;
  std::cout << "g " << g << std::endl;
  std::cout << "r " << r.transpose() << std::endl;
  std::cout << "Delta " << Delta << std::endl;
  std::cout << "diag " << diag.transpose() << std::endl;

  std::cout << std::endl << std::endl;

  for (int i = 0; i != 10; ++i)
  {
    std::cout << "iteration " << i << std::endl;

    // calculate parameter (for now 1 / trust region)
    double lambda = 1. / Delta;
    std::cout << "calculated parameter " << lambda << std::endl;

    // set up least squares problem
    Eigen::Matrix<double, num_res + G::lie_dof, G::lie_dof> lhs;
    lhs.template block<G::lie_dof, G::lie_dof>(0, 0) = J;
    lhs.template block<G::lie_dof, G::lie_dof>(G::lie_dof, 0) = sqrt(lambda) * diag.asDiagonal();

    Eigen::Matrix<double, num_res + G::lie_dof, 1> rhs;
    rhs.template head<num_res>() = -r;
    rhs.template tail<G::lie_dof>().setZero();

    // solve least-squares problem with a big hammer
    Tangent a_hammer = lhs.colPivHouseholderQr().solve(rhs);

    // solve least-squares problem with a small scalpel
    Eigen::ColPivHouseholderQR<Eigen::Matrix<double, G::lie_dof, G::lie_dof>> J_qrdecomp(J);
    Tangent a = solve_ls<G::lie_dof>(J_qrdecomp, sqrt(lambda) * diag, r);

    std::cout << "step       " << a.transpose() << std::endl;
    std::cout << "step truth " << a_hammer.transpose() << std::endl;

    double rpre_n = r.stableNorm();

    // update optimization variables
    g += a;
    J = f.df(g);
    r = f(g);
    std::cout << "new g " << g << std::endl;
    std::cout << "new r " << r.transpose() << std::endl;

    // calculate actual to predicted reduction
    double up = 1. - Eigen::numext::abs2(r.stableNorm() / rpre_n);
    double dn1 = Eigen::numext::abs2((J * a).stableNorm() / rpre_n);
    double dn2 = 2 * Eigen::numext::abs2(std::sqrt(lambda) * (diag.cwiseProduct(a)).stableNorm() / rpre_n);
    double rho = up / (dn1 + dn2);
    std::cout << "rho " << rho << std::endl;

    // update trust region
    if (rho < 0.25) {
      Delta /= 2;
    } else if ((rho <= 0.75 && lambda == 0) || lambda >= 0.75) {
      Delta = 2 * diag.cwiseProduct(a).stableNorm();
    }
    std::cout << "updated Delta " << Delta << std::endl;


    // update scaling
    const auto J_n = J.colwise().stableNorm().transpose().eval();
    diag = diag.cwiseMax(J_n);
    std::cout << "updated diag " << diag.transpose() << std::endl;

    std::cout << std::endl << std::endl;
  }
}

#endif  // SMOOTH__OPTIM__LM_HPP_
