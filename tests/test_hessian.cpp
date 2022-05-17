// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#ifdef ENABLE_AUTODIFF_TESTS
#include <smooth/compat/autodiff.hpp>
#endif

#include <smooth/derivatives.hpp>
#include <smooth/diff.hpp>
#include <smooth/se2.hpp>
#include <smooth/se3.hpp>

template<smooth::LieGroup G>
struct Functor
{
  G xd;

  Functor(const G & g) : xd(g) {}

  // make sure we can differentiate without making copies..
  Functor(const Functor &) = delete;
  Functor & operator=(const Functor &) = delete;
  Functor(Functor &&)                  = delete;
  Functor & operator=(Functor &&) = delete;

  template<typename Scalar>
  Scalar operator()(const smooth::CastT<Scalar, G> & x) const
  {
    return (x - xd.template cast<Scalar>()).squaredNorm() / Scalar(2);
  }

  Eigen::RowVector<double, smooth::Dof<G>> jacobian(const smooth::CastT<double, G> & x) const
  {
    const auto e = rminus(x, xd);
    return smooth::dr_rminus_squarednorm<G>(e);
  }

  template<typename Scalar>
  Eigen::Vector<Scalar, smooth::Dof<G>> jacobian_ad(const smooth::CastT<Scalar, G> & x) const
  {
    const auto e = rminus(x, xd);
    return smooth::dr_rminus_squarednorm<G>(e.template cast<Scalar>());
  }

  Eigen::Matrix<double, smooth::Dof<G>, smooth::Dof<G>> hessian(const G & x) const
  {
    const auto e = rminus(x, xd);
    return smooth::d2r_rminus_squarednorm<G>(e);
  }
};

TEST(Hessian, RminusSE2)
{
  using G = smooth::SE2d;

  for (auto i = 0u; i < 10; ++i) {
    const auto f = Functor<G>{G::Random()};
    const auto x = G::Random();

    const auto [f_num, drf_num, d2f_num] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));

    const auto [f_anal, drf_anal, d2f_anal] =
      smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));

    const auto [j, dj_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
      [&f](const auto & var) { return f.jacobian_ad(var); }, smooth::wrt(x));

    ASSERT_TRUE(j.transpose().isApprox(drf_num, 1e-3));
    ASSERT_TRUE(drf_anal.isApprox(drf_num, 1e-3));

    ASSERT_TRUE(dj_num.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(d2f_anal.isApprox(d2f_num, 1e-3));

#ifdef ENABLE_AUTODIFF_TESTS
    const auto [f_ad, drf_ad, d2f_ad] =
      smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));
    ASSERT_TRUE(d2f_ad.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(drf_ad.isApprox(drf_num, 1e-3));
#endif
  }
}

TEST(Hessian, RminusSE3)
{
  using G = smooth::SE3d;

  for (auto i = 0u; i < 10; ++i) {
    const auto f = Functor<G>{G::Random()};
    const auto x = G::Random();

    const auto [f_num, drf_num, d2f_num] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));

    const auto [f_anal, drf_anal, d2f_anal] =
      smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));

    const auto [j, dj_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
      [&f](const auto & var) { return f.jacobian_ad(var); }, smooth::wrt(x));

    ASSERT_TRUE(j.transpose().isApprox(drf_num, 1e-3));
    ASSERT_TRUE(drf_anal.isApprox(drf_num, 1e-3));

    ASSERT_TRUE(dj_num.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(d2f_anal.isApprox(d2f_num, 1e-3));

#ifdef ENABLE_AUTODIFF_TESTS
    const auto [f_ad, drf_ad, d2f_ad] =
      smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));
    ASSERT_TRUE(d2f_ad.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(drf_ad.isApprox(drf_num, 1e-3));
#endif
  }
}

TEST(Hessian, ScalarFunc)
{
  const auto fun = []<typename T>(Eigen::VectorX<T> x) -> Eigen::VectorX<T> {
    return (x * x.transpose()).colwise().sum();
  };

  const auto fun1 = [&fun](Eigen::VectorXd x) -> double { return fun(x)(0); };
  const auto fun2 = [&fun](Eigen::VectorXd x) -> double { return fun(x)(1); };
  const auto fun3 = [&fun](Eigen::VectorXd x) -> double { return fun(x)(2); };
  const auto fun4 = [&fun](Eigen::VectorXd x) -> double { return fun(x)(3); };
  const auto fun5 = [&fun](Eigen::VectorXd x) -> double { return fun(x)(4); };

  Eigen::VectorXd x = Eigen::VectorXd::Random(5);

  const auto [f, df, d2f] = smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun, smooth::wrt(x));

  const auto [f1, df1, d2f1] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun1, smooth::wrt(x));
  const auto [f2, df2, d2f2] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun2, smooth::wrt(x));
  const auto [f3, df3, d2f3] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun3, smooth::wrt(x));
  const auto [f4, df4, d2f4] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun4, smooth::wrt(x));
  const auto [f5, df5, d2f5] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(fun5, smooth::wrt(x));

  ASSERT_EQ(d2f.rows(), 5);
  ASSERT_EQ(d2f.cols(), 25);

  ASSERT_TRUE(d2f1.isApprox(d2f.middleCols(0, 5)));
  ASSERT_TRUE(d2f2.isApprox(d2f.middleCols(5, 5)));
  ASSERT_TRUE(d2f3.isApprox(d2f.middleCols(10, 5)));
  ASSERT_TRUE(d2f4.isApprox(d2f.middleCols(15, 5)));
  ASSERT_TRUE(d2f5.isApprox(d2f.middleCols(20, 5)));

#ifdef ENABLE_AUTODIFF_TESTS
  const auto [f_ad, df_ad, d2f_ad] =
    smooth::diff::dr<2, smooth::diff::Type::Autodiff>(fun, smooth::wrt(x));
  ASSERT_TRUE(d2f_ad.isApprox(d2f, 1e-3));
#endif
}

TEST(Hessian, Rminus_full)
{
  using G = smooth::SE3d;

  const auto x = G::Random(), y = G::Random();

  const auto f = [&y]<typename T>(smooth::SE3<T> xvar) -> Eigen::Vector3<T> {
    return (xvar - y.template cast<T>()).template head<3>();
  };

  const auto [f_num, df_num, d2f_num] =
    smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));

  ASSERT_EQ(df_num.rows(), 3);
  ASSERT_EQ(df_num.cols(), 6);
  ASSERT_EQ(d2f_num.rows(), 6);
  ASSERT_EQ(d2f_num.cols(), 3 * 6);

#ifdef ENABLE_AUTODIFF_TESTS
  const auto [f_ad, df_ad, d2f_ad] =
    smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));
  ASSERT_TRUE(f_ad.isApprox(f_num, 1e-3));
  ASSERT_TRUE(df_ad.isApprox(df_num, 1e-3));
  ASSERT_TRUE(d2f_ad.isApprox(d2f_num, 1e-3));
#endif
}

TEST(Hessian, rminus)
{
  using G = smooth::SE2d;

  for (auto i = 0u; i < 5; ++i) {
    const G x = G::Random(), y = G::Random();

    const auto [unused1, unused2, H_num] = smooth::diff::dr<2, smooth::diff::Type::Numerical>(
      [&y](const auto & var) -> smooth::Tangent<G> { return rminus(var, y); }, smooth::wrt(x));

    const auto H_ana = smooth::d2r_rminus<G>(rminus(x, y));

    ASSERT_TRUE(H_num.isApprox(H_ana, 1e-4));
  }
}
