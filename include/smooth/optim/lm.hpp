#ifndef SMOOTH__OPTIM__LM_HPP_
#define SMOOTH__OPTIM__LM_HPP_

#include <Eigen/Core>
#include <Eigen/Jacobi>

#include <iostream>



// Solve the least-squares problem
//
// min_a  || [ J; diag(d) ] a + [r; 0] ||^2
//
template<int nx>
Eigen::Matrix<double, nx, 1> solve_ls(
  const Eigen::Matrix<double, nx, nx> & J,
  const Eigen::Matrix<double, nx, 1> & d,
  const Eigen::Matrix<double, nx, 1> & r
) {
  using std::sqrt;
  // decomose Q

  std::cout << "Doing QR decomp" << std::endl;
  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, nx, nx>> Jqr(J);

  std::cout << "Result R" << std::endl << Jqr.matrixR();
  std::cout << "Result P" << std::endl << Jqr.colsPermutation().toDenseMatrix() << std::endl;

  // want QR decomposition of matrix
  // [ R  ]
  // [ P' D P ]
  // R is upper diagonal
  // lower part is diagonal

  // need to QR decompose M using givens rotations
  // just need to eliminate lower part
  Eigen::Matrix<double, nx, nx> upper;
  upper.setZero();  // TODO remove
  Eigen::Matrix<double, nx, nx> lower;
  upper.template triangularView<Eigen::Upper>() = Jqr.matrixR().template triangularView<Eigen::Upper>();

  lower = (Jqr.colsPermutation().transpose() * d).asDiagonal();

  Eigen::Matrix<double, nx, 1> QTr = Jqr.matrixQ().transpose() * r;

  std::cout << "Starting elimintation" << std::endl;
  std::cout << "upper" << std::endl << upper << std::endl;
  std::cout << "lower" << std::endl << lower << std::endl;

  Eigen::Matrix<double, 2 * nx, 2 * nx> QQ;
  Eigen::Matrix<double, 2 * nx, nx> RR, RR0;
  QQ.setIdentity();
  RR.template topLeftCorner<nx, nx>() = upper;
  RR.template bottomLeftCorner<nx, nx>() = lower;

  RR0 = RR;

  std::cout << RR << std::endl;

  Eigen::JacobiRotation<double> rot;
  for (auto i = 0u; i != nx; ++i)  // for each column
  {
    // find permuted d element
    std::cout << "finding permuted element" << std::endl;
    if (lower(i, i) == 0.) {
      continue;
    }

    std::cout << "eliminating column " << i << std::endl;

    // eliminate all elements in lower part
    for (int k = i; k >= 0; --k) {  // for each row
      rot.makeGivens(RR(i, i), RR(nx + k, i));

      Eigen::Matrix<double, 2 * nx, 2 * nx> rotmat;
      rotmat.setIdentity();
      rotmat(i, i) = rot.c();
      rotmat(i, nx + k) = rot.s();
      rotmat(nx + k, i) = -rot.s();
      rotmat(nx + k, nx + k) = rot.c();

      std::cout << "k = " << k << std::endl;
      std::cout << rotmat << std::endl;

      RR = rotmat.transpose() * RR;
      QQ = QQ * rotmat;
    }

    std::cout << "after elimination" << std::endl;

    std::cout << RR << std::endl;
  }

  std::cout << "started with" << std::endl;
  std::cout << RR0 << std::endl;
  std::cout << "result" << std::endl;
  std::cout << QQ * RR << std::endl;

  QTr = QQ.template topLeftCorner<nx, nx>().transpose() * QTr;
  RR.template topLeftCorner<nx, nx>().template triangularView<Eigen::Upper>().solveInPlace(QTr);

  return Jqr.colsPermutation() * QTr;
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

    // solve least-squares problem
    Tangent a = lhs.colPivHouseholderQr().solve(rhs);
    Tangent a2 = solve_ls<G::lie_dof>(J, sqrt(lambda) * diag, -r);

    std::cout << "step " << a.transpose() << std::endl;
    std::cout << "step2 " << a2.transpose() << std::endl;

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
