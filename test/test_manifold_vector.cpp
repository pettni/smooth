#include <gtest/gtest.h>

#include <sstream>

#include "smooth/manifold_vector.hpp"
#include "smooth/so3.hpp"
#include "smooth/nls.hpp"

template<smooth::Manifold M>
void test(const M &)
{}

TEST(ManifoldVector, Static)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  test(m);
}

TEST(ManifoldVector, Construct)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m1, m2;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());

  auto log = m2 - m1;

  ASSERT_EQ(log.size(), 9);
}

TEST(ManifoldVector, Cast)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m1;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  auto m1_cast = m1.cast<float>();

  ASSERT_EQ(m1_cast.size(), 9);
}

TEST(ManifoldVector, Optimize)
{
  auto f = [] (const auto & var) {
    using Scalar = typename std::decay_t<decltype(var)>::Scalar;
    Eigen::Matrix<Scalar, 3, 1> ret;
    ret.setZero();
    for (const auto & gi : var) {
      ret += gi.log().cwiseAbs2();
    }
    return ret;
  };

  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  smooth::minimize(f, m);

  for (auto x : m) {
    ASSERT_LE(x.log().norm(), 1e-5);
  }
}

TEST(ManifoldVector, print)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  std::stringstream ss;
  ss << m << std::endl;
  ASSERT_GE(ss.str().size(), 0);
}

