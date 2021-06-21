#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>

#include "bundle.hpp"
#include "so3.hpp"
#include "tn.hpp"

template<typename G>
void test(G g)
{
  // test explog
  auto l      = g.log();
  auto g_copy = G::exp(l);
  std::cout << "explog " << g_copy.coeffs().isApprox(g.coeffs()) << std::endl;

  auto g_cast = g.template cast<float>();
  std::cout << "cast " << g_cast << std::endl;

  // test maps
  std::array<double, G::RepSize> d1, d2;
  Eigen::Map<G> gm1(d1.data()), gm2(d2.data());
  gm1 = g;
  gm2 = gm1;
  G g2;
  g2 = gm2;

  std::cout << "copy circle " << g2.coeffs().isApprox(g.coeffs()) << std::endl;

  Eigen::Map<const Eigen::Matrix<double, G::RepSize, 1>> mm(g2.data());
  std::cout << "map of maps " << mm.transpose() << std::endl;

  // set identity
  g.setIdentity();

  G g3 = g2.inverse();
  std::cout << "inverse " << (g2 * g3).coeffs().isApprox(g.coeffs()) << std::endl;

  auto x = g3.Ad();
  std::cout << x << std::endl;
}

int main()
{
  using SO3d     = smooth::SO3<double>;
  using T4d      = smooth::Tn<4, double>;
  using MyBundle = smooth::Bundle<SO3d, SO3d, T4d, T4d>;

  SO3d g(Eigen::Quaterniond(1, 2, 3, 4).normalized());

  std::cout << "TESTING SO3" << std::endl;
  test(g);

  MyBundle b;
  b.part<0>() = SO3d(Eigen::Quaterniond::UnitRandom());
  b.part<1>() = SO3d(Eigen::Quaterniond::UnitRandom());
  b.part<2>().setRandom();
  b.part<3>().setRandom();
  std::cout << "TESTING BUNDLE" << std::endl;
  test(b);

  return 0;
}
