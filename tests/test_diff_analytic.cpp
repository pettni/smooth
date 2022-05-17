// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/diff.hpp"

struct Functor
{
  Eigen::Matrix<double, 5, 5> hess{};

  double operator()(const Eigen::Vector2d &) const { return 0.; }

  Eigen::RowVector<double, 5> jacobian(const Eigen::Vector2d &) const
  {
    return Eigen::RowVector<double, 5>::Zero();
  }

  std::reference_wrapper<const Eigen::Matrix<double, 5, 5>> hessian(const Eigen::Vector2d &)
  {
    hess.setConstant(1);
    return hess;
  }
};

TEST(DiffAnalytic, ReturnType)
{
  Functor f{};

  Eigen::Vector2d x = Eigen::Vector2d::Random();

  auto [fv1, dfv1] = smooth::diff::dr<1, smooth::diff::Type::Analytic>(f, smooth::wrt(x));
  static_assert(std::is_same_v<decltype(fv1), double>);
  static_assert(std::is_same_v<decltype(dfv1), Eigen::RowVector<double, 5>>);

  auto [fv2, dfv2, d2fv2] = smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));
  static_assert(std::is_same_v<decltype(fv2), double>);
  static_assert(std::is_same_v<decltype(dfv2), Eigen::RowVector<double, 5>>);
  static_assert(std::is_same_v<decltype(d2fv2), const Eigen::Matrix<double, 5, 5> &>);
}
