#include <gtest/gtest.h>

#include "smooth/common.hpp"
#include "smooth/manifold_vector.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

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
  using G = smooth::ManifoldVector<double, smooth::SO3>;
  G g;

  g.push_back(smooth::SO3d::Random());
  g.push_back(smooth::SO3d::Random());
  g.push_back(smooth::SO3d::Random());
}
