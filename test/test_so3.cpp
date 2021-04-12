#include <gtest/gtest.h>

#include <unsupported/Eigen/MatrixFunctions>  // for matrix exponential

#include "smooth/so3.hpp"

#include "reverse_storage.hpp"


TEST(SO3Interface, Quaternion)
{
  static_assert(smooth::SO3d::size == 4);
  static_assert(smooth::SO3d::dof == 3);
  static_assert(smooth::SO3d::dim == 3);

  // test unordered unit quaternion
  std::default_random_engine rng(5);
  smooth::SO3d g = smooth::SO3d::Random(rng);
  smooth::SO3<double, smooth::ReverseStorage<double, 4>> g_rev(g);

  const auto q = g_rev.unit_quaternion();

  for (auto i = 0u; i != 4; ++i)
  {
    ASSERT_DOUBLE_EQ(q.coeffs()[i], g_rev.coeffs().a[3 - i]);
  }

  ASSERT_TRUE(q.isApprox(g.unit_quaternion()));
}
