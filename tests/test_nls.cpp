// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include "smooth/optim.hpp"
#include "smooth/so3.hpp"

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
  opts.ftol    = 1e-12;
  opts.ptol    = 1e-12;
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
  opts.ftol    = 1e-12;
  opts.ptol    = 1e-12;
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
  Eigen::VectorX<T> operator()(const smooth::SO3<T> & g1, const smooth::SO3<T> & g2, const smooth::SO3<T> & g3)
  {
    Eigen::VectorX<T> f(9);
    f.template segment<3>(0) = g1.log();
    f.template segment<3>(3) = (g3 - g2) - d23;
    f.template segment<3>(6) = (g1 - g3) - d31;
    return f;
  }

  Eigen::SparseMatrix<double> jacobian(const smooth::SO3d & g1, const smooth::SO3d & g2, const smooth::SO3d & g3) const
  {
    const Eigen::Matrix3d dr_f1_g1 = smooth::SO3d::dr_expinv(g1.log());

    const Eigen::Matrix3d dr_f2_g3 = smooth::SO3d::dr_expinv(g3 - g2);
    const Eigen::Matrix3d dr_f2_g2 = -smooth::SO3d::dl_expinv(g3 - g2);

    const Eigen::Matrix3d dr_f3_g1 = smooth::SO3d::dr_expinv(g1 - g3);
    const Eigen::Matrix3d dr_f3_g3 = -smooth::SO3d::dl_expinv(g1 - g3);

    Eigen::SparseMatrix<double> dr_f;
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

    return dr_f;
  }

  Eigen::Vector3d d23, d31;
};

TEST(NLS, AnalyticSparse)
{
  auto f = AnalyticSparseFunctor{Eigen::Vector3d::Random(), Eigen::Vector3d::Random()};

  smooth::SO3d g1a, g2a, g3a;
  g1a.setRandom();
  g2a.setRandom();
  g3a.setRandom();

  smooth::SO3d g1b = g1a, g2b = g2a, g3b = g3a;
  smooth::SO3d g1c = g1a, g2c = g2a, g3c = g3a;

  // check that we are differentiable
  static_assert(smooth::diff::detail::diffable_order1<decltype(f), decltype(smooth::wrt(g1a, g2a, g3a))> == true);
  static_assert(smooth::diff::detail::diffable_order2<decltype(f), decltype(smooth::wrt(g1a, g2a, g3a))> == false);

  // solve with analytic diff
  smooth::minimize<smooth::diff::Type::Analytic>(f, smooth::wrt(g1a, g2a, g3a));

  // solve with numerical autodiff
  smooth::minimize<smooth::diff::Type::Numerical>(f, smooth::wrt(g1b, g2b, g3b));

  // solve with default autodiff
  smooth::minimize<smooth::diff::Type::Default>(f, smooth::wrt(g1c, g2c, g3c));

  ASSERT_TRUE(g1a.isApprox(g1b, 1e-5));
  ASSERT_TRUE(g2a.isApprox(g2b, 1e-5));
  ASSERT_TRUE(g3a.isApprox(g3b, 1e-5));

  ASSERT_TRUE(g1a.isApprox(g1c, 1e-5));
  ASSERT_TRUE(g2a.isApprox(g2c, 1e-5));
  ASSERT_TRUE(g3a.isApprox(g3c, 1e-5));
}
