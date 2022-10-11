// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <iostream>

#include <smooth/diff.hpp>
#include <smooth/optim.hpp>
#include <smooth/so3.hpp>

void diff()
{
  std::srand(10);

  // mapping \f$ G \times G \rightarrow R^n \f$
  auto f = []<typename T>(const smooth::SO3<T> & v1, const smooth::SO3<T> & v2) { return (v1 * v2).log(); };

  smooth::SO3d g1 = smooth::SO3d::Random();
  smooth::SO3d g2 = smooth::SO3d::Random();

  // differentiate f at (g1, g2) w.r.t. first argument
  auto [fval1, J1] = smooth::diff::dr<1>(f, smooth::wrt(g1, g2), std::index_sequence<0>{});

  // differentiate f at (g1, g2) w.r.t. second argument
  auto [fval2, J2] = smooth::diff::dr<1>(f, smooth::wrt(g1, g2), std::index_sequence<1>{});

  // differentiate f at (g1, g2) w.r.t. both arguments
  auto [fval, J] = smooth::diff::dr<1>(f, smooth::wrt(g1, g2));

  // Now J == [J1, J2]

  std::cout << J << '\n';
  std::cout << J1 << '\n';
  std::cout << J2 << '\n';
}

void optim()
{
  std::srand(10);

  smooth::SO3d g1       = smooth::SO3d::Random();
  const smooth::SO3d g2 = smooth::SO3d::Random();

  // function defining residual
  auto f = [&g2]<typename T>(const smooth::SO3<T> & v1) { return (v1 * g2.template cast<T>()).log(); };

  // minimize || f ||^2 w.r.t. g1 (g1 is modified in-place)
  smooth::minimize(f, smooth::wrt(g1));

  // Now g1 == g2.inverse()
  std::cout << g1 << '\n';
  std::cout << g2 << '\n';
}

int main()
{
  std::cout << "RUNNING DIFF" << '\n';
  diff();

  std::cout << "RUNNING OPTIM" << '\n';
  optim();

  return EXIT_SUCCESS;
}
