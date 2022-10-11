// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <sstream>

#include <gtest/gtest.h>

#include "smooth/manifolds.hpp"
#include "smooth/optim.hpp"
#include "smooth/so3.hpp"

template<smooth::Manifold M>
void test(const M &)
{
  M m = smooth::Default<M>(3);
  ASSERT_EQ(smooth::dof(m), 3);
}

TEST(ManifoldVector, Static)
{
  std::vector<smooth::SO3d> m1;
  std::vector<Eigen::VectorXd> m2;

  test(m1);
  test(m2);
}

TEST(ManifoldVector, Construct)
{
  using M = std::vector<smooth::SO3d>;
  M m1, m2;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());

  auto log = smooth::rminus(m2, m1);

  ASSERT_EQ(log.size(), 9);
}

TEST(ManifoldVector, Dynamic)
{
  using M = std::vector<Eigen::VectorXd>;
  M m1, m2;

  m1.push_back(Eigen::VectorXd::Random(3));
  m1.push_back(Eigen::VectorXd::Random(4));
  m1.push_back(Eigen::VectorXd::Random(2));

  m2.push_back(Eigen::VectorXd::Random(3));
  m2.push_back(Eigen::VectorXd::Random(4));
  m2.push_back(Eigen::VectorXd::Random(2));

  auto log = smooth::rminus(m2, m1);

  auto plus = smooth::rplus(m1, log);

  ASSERT_EQ(log.size(), 9);
  ASSERT_EQ(smooth::dof(plus), 9);
}

TEST(ManifoldVector, Cast)
{
  using M = std::vector<smooth::SO3d>;
  M m1;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  auto m1_cast = smooth::cast<float>(m1);

  ASSERT_EQ(m1_cast.size(), 3);
  ASSERT_EQ(smooth::dof(m1_cast), 9);
}

TEST(ManifoldVector, Optimize)
{
  auto f = []<typename T>(const std::vector<smooth::SO3<T>> & var) -> Eigen::Vector3<T> {
    Eigen::Vector3<T> ret;
    ret.setZero();
    for (const auto & gi : var) { ret += gi.log().cwiseAbs2(); }
    return ret;
  };

  using M = std::vector<smooth::SO3d>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  smooth::minimize(f, smooth::wrt(m));

  for (auto x : m) { ASSERT_LE(x.log().norm(), 1e-5); }
}
