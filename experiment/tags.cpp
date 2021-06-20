#include <Eigen/Core>

#include <iostream>

#include "so3.hpp"
#include "tn.hpp"
#include "bundle.hpp"

int main()
{
  using SO3d     = SO3<double>;
  using T4d      = T<4, double>;
  using MyBundle = Bundle<double, SO3Tag, SO3Tag, TnTag<2>>;

  SO3d g;
  MyBundle b;
  T4d t;

  auto l1 = g.log();
  auto l2 = b.log();
  auto l3 = t.log();

  auto s1 = SO3d::exp(Eigen::Matrix<double, 3, 1>::Zero());
  auto s2 = MyBundle::exp(Eigen::Matrix<double, 8, 1>::Zero());
  auto s3 = T4d::exp(Eigen::Matrix<double, 4, 1>::Zero());

  auto c1 = g.cast<float>();
  auto c2 = b.cast<float>();
  auto c3 = t.cast<float>();

  std::array<double, 4> d;
  SO3Map<double> gm(d.data());
  gm = g;
  auto se2mc = gm.cast<float>();

  g = gm;

  std::cout << g << std::endl;
  std::cout << gm << std::endl;
  std::cout << se2mc << std::endl;

  return 0;
}
