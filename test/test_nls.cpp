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

#include <gtest/gtest.h>

#include <Eigen/Sparse>

#include "smooth/lie_group.hpp"
#include "smooth/optim.hpp"
#include "smooth/so3.hpp"

template<int N, int M>
void run_leastsquares_test(bool zero_d, bool sing)
{
  // static
  for (auto i = 0u; i != 10; ++i) {
    Eigen::Matrix<double, M, N> J;
    Eigen::Matrix<double, M, 1> r;
    Eigen::Matrix<double, N, 1> d;

    J.setRandom();
    if (sing) {
      J.col(N / 2).setZero();
      J.row(M / 2).setZero();
    }

    d.setRandom();
    d = (d + Eigen::Matrix<double, N, 1>::Ones()).cwiseMax(0);
    if (zero_d) { d.setZero(); }

    r.setRandom();

    // solve static
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto a1 = smooth::detail::solve_ls<N, M>(J_qr, d, r);

    // solve dynamic
    Eigen::Matrix<double, -1, -1> Jd = J;
    Eigen::Matrix<double, -1, 1> rd  = r;
    Eigen::Matrix<double, -1, 1> dd  = d;
    Eigen::ColPivHouseholderQR<decltype(Jd)> Jd_qr(Jd);
    auto a2 = smooth::detail::solve_ls<-1, -1>(Jd_qr, dd, rd);

    // solve sparse
    Eigen::SparseMatrix<double> Jsp;
    Jsp = J.sparseView();
    Eigen::SparseQR<decltype(Jsp), Eigen::COLAMDOrdering<int>> Jsp_qr(Jsp);
    auto a3 = smooth::detail::solve_ls<-1, -1>(Jsp_qr, dd, rd);

    // solve for precise solution
    Eigen::Matrix<double, N + M, N> lhs;
    lhs.template topLeftCorner<M, N>()    = J;
    lhs.template bottomLeftCorner<N, N>() = d.asDiagonal();

    Eigen::Matrix<double, N + M, 1> rhs;
    rhs.template head<M>() = -r;
    rhs.template tail<N>().setZero();
    Eigen::Matrix<double, N, 1> a_verif = lhs.fullPivHouseholderQr().solve(rhs);

    // verify that all methods gave the same result
    ASSERT_TRUE(a1.isApprox(a_verif));
    ASSERT_TRUE(a2.isApprox(a_verif));
    ASSERT_TRUE(a3.isApprox(a_verif));
  }
}

TEST(Optimization, LeastSquares)
{
  run_leastsquares_test<1, 1>(false, false);
  run_leastsquares_test<5, 1>(false, false);
  run_leastsquares_test<5, 10>(false, false);
  run_leastsquares_test<8, 16>(false, false);

  run_leastsquares_test<1, 1>(false, true);
  run_leastsquares_test<5, 1>(false, true);
  run_leastsquares_test<5, 10>(false, true);
  run_leastsquares_test<8, 16>(false, true);

  run_leastsquares_test<1, 1>(true, false);
  run_leastsquares_test<5, 10>(true, false);
  run_leastsquares_test<8, 16>(true, false);

  run_leastsquares_test<5, 10>(true, true);
  run_leastsquares_test<8, 16>(true, true);
}

TEST(Optimization, LmPar)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 1;

  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    // solve static
    auto [par1, x] = smooth::detail::lmpar<4, 4>(J, d, r, Delta);

    // solve dynamic
    Eigen::MatrixXd Jd = J;
    Eigen::VectorXd rd = r;
    Eigen::VectorXd dd = d;
    auto [par2, xd]    = smooth::detail::lmpar<-1, -1>(Jd, dd, rd, Delta);

    // solve sparse1
    Eigen::SparseMatrix<double> Jsp;
    Jsp               = J.sparseView();
    auto [par3, xsp1] = smooth::detail::lmpar<-1, -1>(Jsp, dd, rd, Delta);

    // solve sparse2
    auto [par4, xsp2] = smooth::detail::lmpar_sparse(Jsp, dd, rd, Delta);

    // check equality of static and dynamic
    ASSERT_NEAR(par1, par2, 1e-10);
    ASSERT_NEAR(par1, par3, 1e-10);
    ASSERT_NEAR(par1, par4, 1e-10);
    ASSERT_TRUE(x.isApprox(xd));
    ASSERT_TRUE(x.isApprox(xsp1));
    ASSERT_TRUE(x.isApprox(xsp2));

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = smooth::detail::solve_ls<N, N>(J_qr, sqrt(par1) * d, r);
    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par1 == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par1 > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);
  }
}

TEST(Optimization, LmParSmall)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 0.1;
  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    // solve static
    auto [par1, x] = smooth::detail::lmpar<4, 4>(J, d, r, Delta);

    // solve dynamic
    Eigen::MatrixXd Jd = J;
    Eigen::VectorXd rd = r;
    Eigen::VectorXd dd = d;
    auto [par2, xd]    = smooth::detail::lmpar<-1, -1>(Jd, dd, rd, Delta);

    // solve sparse
    Eigen::SparseMatrix<double> Jsp;
    Jsp               = J.sparseView();
    auto [par3, xsp1] = smooth::detail::lmpar<-1, -1>(Jsp, dd, rd, Delta);

    // solve sparse2
    auto [par4, xsp2] = smooth::detail::lmpar_sparse(Jsp, dd, rd, Delta);

    // check equality of static and dynamic
    ASSERT_NEAR(par1, par2, 1e-10);
    ASSERT_NEAR(par1, par3, 1e-10);
    ASSERT_NEAR(par1, par4, 1e-10);
    ASSERT_TRUE(x.isApprox(xd));
    ASSERT_TRUE(x.isApprox(xsp1));
    ASSERT_TRUE(x.isApprox(xsp2));

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = smooth::detail::solve_ls<N, N>(J_qr, sqrt(par1) * d, r);
    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par1 == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par1 > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);
  }
}

TEST(Optimization, LmParSing)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 1;
  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();
    J.col(3).setZero();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    // solve QR
    auto [par1, x] = smooth::detail::lmpar<4, 4>(J, d, r, Delta);

    // solve dynamic
    Eigen::MatrixXd Jd = J;
    Eigen::VectorXd rd = r;
    Eigen::VectorXd dd = d;
    auto [par2, xd]    = smooth::detail::lmpar<-1, -1>(Jd, dd, rd, Delta);

    // solve sparse1
    Eigen::SparseMatrix<double> Jsp;
    Jsp               = J.sparseView();
    auto [par3, xsp1] = smooth::detail::lmpar<-1, -1>(Jsp, dd, rd, Delta);

    // solve sparse2
    auto [par4, xsp2] = smooth::detail::lmpar_sparse(Jsp, dd, rd, Delta);

    // check equality of static and dynamic
    ASSERT_NEAR(par1, par2, 1e-10);
    ASSERT_NEAR(par1, par3, 1e-10);
    ASSERT_TRUE(x.isApprox(xd));
    ASSERT_TRUE(x.isApprox(xsp1));

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = smooth::detail::solve_ls<N, N>(J_qr, sqrt(par1) * d, r);

    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par1 == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par1 > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);

    // don't expect sparse2 to be equal
    // check that parameter satisfies conditions
    bool cond3 = (par4 == 0) && (d.asDiagonal() * xsp2).norm() <= 1.1 * Delta;
    bool cond4 = (par4 > 0) && std::abs((d.asDiagonal() * xsp2).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond3 || cond4);
  }
}

TEST(NLS, MultipleArgsStatic)
{
  smooth::SO3d g1, g2;
  g1.setRandom();
  g2.setRandom();

  auto f = [](auto v1, auto v2) {
    Eigen::Vector3d diff = (v1 - v2) - Eigen::Vector3d::Ones();
    Eigen::Matrix<double, 9, 1> ret;
    ret << v1.log(), v2.log(), diff;
    return ret;
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;
  opts.verbose = true;

  smooth::minimize(f, smooth::wrt(g1, g2), opts);

  ASSERT_TRUE(g1.inverse().isApprox(g2, 1e-6));
}

TEST(NLS, MultipleArgsDynamic)
{
  smooth::SO3d g1, g2;
  g1.setRandom();
  g2.setRandom();

  auto f = [](auto v1, auto v2) -> Eigen::VectorXd {
    Eigen::VectorXd diff = (v1 - v2) - Eigen::Vector3d::Ones();
    Eigen::Matrix<double, 9, 1> ret;
    ret << v1.log(), v2.log(), diff;
    return ret;
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;
  opts.verbose = true;

  smooth::minimize(f, smooth::wrt(g1, g2), opts);

  ASSERT_TRUE(g1.inverse().isApprox(g2, 1e-6));
}

TEST(NLS, MixedArgs)
{
  smooth::SO3d g0, g1;
  Eigen::VectorXd v(3);
  g0.setRandom();
  g1.setRandom();
  v.setRandom();

  auto f = [&](auto var_g, auto var_vec) -> Eigen::VectorXd {
    Eigen::Matrix<double, -1, 1> ret(6);
    ret << (var_g + var_vec.template head<3>()) - g0, var_vec - Eigen::Vector3d::Ones();
    return ret;
  };

  smooth::minimize(f, smooth::wrt(g1, v));

  auto g1_plus_v = g1 + v.head<3>();
  ASSERT_TRUE(g1_plus_v.isApprox(g0, 1e-6));
  ASSERT_TRUE(v.isApprox(Eigen::Vector3d::Ones(), 1e-6));
}

struct AnalyticSparseFunctor
{
  template<typename T>
  auto operator()(const smooth::SO3<T> & g1, const smooth::SO3<T> & g2, const smooth::SO3<T> & g3)
  {
    auto dr_f1_g1 = smooth::SO3d::dr_expinv(g1.log());

    auto dr_f2_g3 = smooth::SO3d::dr_expinv(g3 - g2);
    auto dr_f2_g2 = (-smooth::SO3d::dl_expinv(g3 - g2)).eval();

    auto dr_f3_g1 = smooth::SO3d::dr_expinv(g1 - g3);
    auto dr_f3_g3 = (-smooth::SO3d::dl_expinv(g1 - g3)).eval();

    Eigen::Matrix<T, -1, 1> f(9);
    f.template segment<3>(0) = g1.log();
    f.template segment<3>(3) = (g3 - g2) - d23;
    f.template segment<3>(6) = (g1 - g3) - d31;

    Eigen::SparseMatrix<T> dr_f;
    dr_f.resize(9, 9);
    for (int i = 0; i != 3; ++i) {
      for (int j = 0; j != 3; ++j) {
        dr_f.insert(i, j) = dr_f1_g1(i, j);

        dr_f.insert(3 + i, 3 + j) = dr_f2_g2(i, j);
        dr_f.insert(3 + i, 6 + j) = dr_f2_g3(i, j);

        dr_f.insert(6 + i, 6 + j) = dr_f3_g3(i, j);
        dr_f.insert(6 + i, 0 + j) = dr_f3_g1(i, j);
      }
    }
    dr_f.makeCompressed();

    return std::make_pair(f, dr_f);
  }

  Eigen::Vector3d d23, d31;
};

TEST(NLS, AnalyticSparse)
{
  auto f = AnalyticSparseFunctor{Eigen::Vector3d::Random(), Eigen::Vector3d::Random()};

  smooth::SO3d g1, g2, g3;
  g1.setRandom();
  g2.setRandom();
  g3.setRandom();

  auto g1c = g1;
  auto g2c = g2;
  auto g3c = g3;

  // solve sparse
  smooth::minimize<smooth::diff::Type::ANALYTIC>(f, smooth::wrt(g1, g2, g3));

  // solve with default
  smooth::minimize<smooth::diff::Type::DEFAULT>(
    [&](auto... var) { return std::get<0>(f(var...)); }, smooth::wrt(g1c, g2c, g3c));

  ASSERT_TRUE(g1.isApprox(g1c, 1e-5));
  ASSERT_TRUE(g2.isApprox(g2c, 1e-5));
  ASSERT_TRUE(g3.isApprox(g3c, 1e-5));
}
