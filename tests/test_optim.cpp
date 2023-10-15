// Copyright (C) 2022 Petter Nilsson. MIT License.

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <gtest/gtest.h>
#include <smooth/diff.hpp>
#include <smooth/optim/tr_solver.hpp>

using namespace smooth;

TEST(TrustRegion, dphi)
{
  static constexpr int N = 5;
  static constexpr int M = 10;

  for (auto iter = 0u; iter < 5; ++iter) {
    const Eigen::MatrixXd Jd = Eigen::MatrixXd::Random(M, N);
    const Eigen::VectorXd d  = Eigen::VectorXd::Random(N).cwiseAbs();
    const Eigen::VectorXd r  = Eigen::VectorXd::Random(M);

    Eigen::SparseMatrix<double> J = Jd.sparseView();

    for (double lambda = 0.; lambda < 20; lambda += 0.1) {
      double dphi;
      const auto x = solve_linear_ldlt(J, d, r, lambda, dphi);

      // test solution validity
      ASSERT_FALSE(x.hasNaN());

      Eigen::MatrixXd H = Jd.transpose() * Jd;
      H.diagonal() += lambda * d.cwiseAbs2();
      ASSERT_LE((H * x + Jd.transpose() * r).cwiseAbs().maxCoeff(), 1e-5);

      // test dphi
      const auto f_diff = [&](double var) -> double {
        const Eigen::VectorXd sol = solve_linear_ldlt(J, d, r, var);
        return d.cwiseProduct(sol).norm();
      };

      const auto [U, dphi_num] = diff::dr<1, diff::Type::Numerical>(f_diff, wrt(lambda));

      ASSERT_NEAR(dphi, dphi_num.x(), 1e-5);
    }
  }
}
