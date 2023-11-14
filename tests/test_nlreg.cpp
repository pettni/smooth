// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "nlreg_data.hpp"
#include "smooth/optim.hpp"

TEST(NlReg, Misra1aStatic)
{
  static constexpr int np   = 2;
  static constexpr int nobs = 14;

  auto [f, data, start1, start2, optim] = Misra1a();

  auto f_vec = [m_f    = std::move(f),
                m_data = std::move(data)](const Eigen::Matrix<double, np, 1> & p) -> Eigen::Matrix<double, nobs, 1> {
    return m_data.col(0).binaryExpr(m_data.col(1), [&](double y, double x) { return m_f(y, x, p); });
  };

  Eigen::Matrix<double, np, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1));
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, np, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2));
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Misra1aDynamic)
{
  auto [f, data, start1, start2, optim] = Misra1a();

  auto f_vec = [m_f    = std::move(f),
                m_data = std::move(data)](const Eigen::Matrix<double, -1, 1> & p) -> Eigen::Matrix<double, -1, 1> {
    return m_data.col(0).binaryExpr(m_data.col(1), [&](double y, double x) { return m_f(y, x, p); });
  };

  Eigen::Matrix<double, -1, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1));
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, -1, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2));
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Kirby2Static)
{
  static constexpr int np   = 5;
  static constexpr int nobs = 151;

  auto [f, data, start1, start2, optim] = Kirby2();

  auto f_vec = [m_f    = std::move(f),
                m_data = std::move(data)](const Eigen::Matrix<double, np, 1> & p) -> Eigen::Matrix<double, nobs, 1> {
    return m_data.col(0).binaryExpr(m_data.col(1), [&](double y, double x) { return m_f(y, x, p); });
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;

  Eigen::Matrix<double, np, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1), opts);
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, np, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2), opts);
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Kirby2Dynamic)
{
  auto [f, data, start1, start2, optim] = Kirby2();

  auto f_vec = [m_f    = std::move(f),
                m_data = std::move(data)](const Eigen::Matrix<double, -1, 1> & p) -> Eigen::Matrix<double, -1, 1> {
    return m_data.col(0).binaryExpr(m_data.col(1), [&](double y, double x) { return m_f(y, x, p); });
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;

  Eigen::Matrix<double, -1, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1), opts);
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, -1, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2), opts);
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}
