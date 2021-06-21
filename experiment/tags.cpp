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

  SO3d g_in;
  g_in.setRandom();
  T4d v_in(1, 2, 3, 4);

  MyBundle b2(g_in, g_in, v_in, 2 * v_in);
  std::cout << b2 << std::endl;

  return 0;
}
