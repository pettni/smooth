#include <gtest/gtest.h>

#include "smooth/common.hpp"
#include "smooth/manifold_vector.hpp"
#include "smooth/so3.hpp"
#include "smooth/nls.hpp"

template<smooth::Manifold M>
void test(const M &)
{}

TEST(ManifoldVector, Static)
{
  using M = smooth::ManifoldVector<double, smooth::SO3>;
  M m;

  test(m);
}

TEST(ManifoldVector, Construct)
{
  using M = smooth::ManifoldVector<double, smooth::SO3>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
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

  using M = smooth::ManifoldVector<double, smooth::SO3>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  smooth::minimize(f, m);

  for (auto x : m) {
    ASSERT_LE(x.log().norm(), 1e-5);
  }
}
