#include <gtest/gtest.h>

#include "smooth/so3.hpp"


TEST(SO3, Copy)
{
  smooth::SO3d so3;
  so3.setIdentity();

  std::array<double, 4> a1{0.5, 0.5, 0.5, 0.5}, a2{0, 0, 0, 1};
  smooth::Map<smooth::SO3d> m1(a1.data()), m2(a2.data());

  so3 = m1;  // map to so3
  m2 = m1;   // map to map

  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, so3.storage(), m1.storage());
  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, m2.storage(), m1.storage());

  so3.setIdentity();

  m1 = so3;  // so3 to map
  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, so3.storage(), m1.storage());
}

TEST(SO3, Constructors)
{
  auto q = Eigen::Quaternionf::UnitRandom();
  smooth::SO3<float> g(q);

  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, g.storage(), q.coeffs());
}
