#include "smooth/interp/bspline_fit.hpp"

#include <gtest/gtest.h>

#include "smooth/se3.hpp"


TEST(BSplineFit, Basic)
{
  std::vector<double> ts{2, 3, 5, 6};
  std::vector<smooth::SO3d> gs{
    smooth::SO3d::Identity(),
    smooth::SO3d::Identity(),
    smooth::SO3d::Identity(),
    smooth::SO3d::Identity()
  };

  smooth::bspline_fit<3, smooth::SO3d>(ts, gs, 0.25);
}
