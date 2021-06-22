#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>

#include "bundle.hpp"
#include "so3.hpp"
#include "so2.hpp"
#include "se2.hpp"
#include "se3.hpp"
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
  gm1 = g;    // storage to map
  gm2 = gm1;  // map to map
  G g2, g3;
  g2 = gm2;   // map to storage
  g3 = g2;    // storage to storage

  G g4(g), g5(gm1);  // copy construct from storage and map

  std::cout << "copy circle " << g3.coeffs().isApprox(g.coeffs()) << std::endl;
  std::cout << "copy circle " << g4.coeffs().isApprox(g.coeffs()) << std::endl;
  std::cout << "copy circle " << g5.coeffs().isApprox(g.coeffs()) << std::endl;

  Eigen::Map<const Eigen::Matrix<double, G::RepSize, 1>> mm(g2.data());
  std::cout << "map of maps " << mm.transpose() << std::endl;

  // set identity
  g.setIdentity();

  G g6 = g2.inverse();
  std::cout << "inverse " << (g2 * g6).coeffs().isApprox(g.coeffs()) << std::endl;

  typename G::TangentMap x = g6.Ad();

  g6 += G::Tangent::UnitX();

  // matrix
  typename G::Matrix M = g6.matrix();

  // hat/vee
  typename G::Tangent a = G::Tangent::Random();

  typename G::Matrix A = G::hat(a);
  typename G::Tangent a2 = G::vee(A);

  std::cout << "hat/vee " << a2.isApprox(a) << std::endl;
}


int main()
{
  using SO2d     = smooth::SO2<double>;
  using SO3d     = smooth::SO3<double>;
  using T4d      = smooth::Tn<4, double>;
  using MyBundle = smooth::Bundle<SO3d, SO3d, T4d, T4d, SO2d>;

  std::cout << "TESTING SO2" << std::endl;
  SO2d so2;
  so2.setRandom();
  test(so2);

  std::cout << "TESTING SO3" << std::endl;
  SO3d so3;
  so3.setRandom();
  test(so3);

  std::cout << "TESTING SE2" << std::endl;
  smooth::SE2<double> se2;
  se2.setRandom();
  test(se2);

  std::cout << "TESTING SE3" << std::endl;
  smooth::SE3<double> se3;
  se3.setRandom();
  test(se3);

  std::cout << "TESTING BUNDLE" << std::endl;
  MyBundle b;
  b.setRandom();
  test(b);

  return 0;
}
