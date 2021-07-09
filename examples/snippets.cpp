#include <smooth/diff.hpp>
#include <smooth/optim.hpp>
#include <smooth/so3.hpp>

#include <iostream>

void diff()
{
  std::srand(10);

  // mapping \f$ G \times G \rightarrow R^n \f$
  auto f = [](auto v1, auto v2) { return (v1 * v2).log(); };

  smooth::SO3d g1 = smooth::SO3d::Random();
  smooth::SO3d g2 = smooth::SO3d::Random();

  // differentiate f at (g1, g2) w.r.t. first argument
  auto [fval1, J1] = smooth::diff::dr(std::bind(f, std::placeholders::_1, g2), smooth::wrt(g1));

  // differentiate f at (g1, g2) w.r.t. second argument
  auto [fval2, J2] = smooth::diff::dr(std::bind(f, g1, std::placeholders::_1), smooth::wrt(g2));

  // differentiate f at (g1, g2) w.r.t. both arguments
  auto [fval, J] = smooth::diff::dr(f, smooth::wrt(g1, g2));

  // Now J == [J1, J2]

  std::cout << J << std::endl;
  std::cout << J1 << std::endl;
  std::cout << J2 << std::endl;
}

void optim()
{
  std::srand(10);

  // mapping \f$ G \times G \rightarrow R^n \f$
  auto f = [](auto v1, auto v2) { return (v1 * v2).log(); };

  smooth::SO3d g1 = smooth::SO3d::Random();
  const smooth::SO3d g2 = smooth::SO3d::Random();

  // minimize f w.r.t. second argument (g1 is modified in-place)
  smooth::minimize(std::bind(f, std::placeholders::_1, g2), smooth::wrt(g1));

  // Now g1 == g2.inverse()
  std::cout << g1 << std::endl;
  std::cout << g2 << std::endl;
}



int main()
{
  std::cout << "RUNNING DIFF" << std::endl;
  diff();

  std::cout << "RUNNING OPTIM" << std::endl;
  optim();

  return EXIT_SUCCESS;
}
