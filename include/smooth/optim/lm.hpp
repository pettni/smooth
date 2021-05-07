#ifndef SMOOTH__OPTIM__LM_HPP_
#define SMOOTH__OPTIM__LM_HPP_

#include <Eigen/Core>

#include <iostream>


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

  for (int i = 0; i != 6; ++i)
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
    std::cout << "step " << a.transpose() << std::endl;

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
