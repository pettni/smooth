#include <gtest/gtest.h>

#include "smooth/interp/bspline.hpp"
#include "smooth/so3.hpp"

TEST(Smooth, BSpline)
{
  std::default_random_engine rng(5);

  std::vector<smooth::SO3d> c;
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));

  typename smooth::SO3d::Tangent vel, acc;

  auto G = smooth::bspline<smooth::SO3d, 3>(c.begin(), c.end(), 0.5, vel, acc);

  std::cout << G << std::endl;
  std::cout << vel << std::endl;
  std::cout << acc << std::endl;
}
