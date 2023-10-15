// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <smooth/optim.hpp>
#include <smooth/so3.hpp>

struct MyFunctor
{
  using Group = smooth::SO3d;

  Eigen::Vector3d operator()(const smooth::SO3d & g) { return g.log(); }

  Eigen::Matrix3d df(const smooth::SO3d & g) { return smooth::SO3d::dr_expinv(g.log()); }
};

int main()
{
  MyFunctor f;
  auto g = smooth::SO3d::Random();
  smooth::MinimizeOptions opts{.verbose = true};
  smooth::minimize(f, smooth::wrt(g), opts);

  return EXIT_SUCCESS;
}
