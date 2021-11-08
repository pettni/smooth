#include <Eigen/Core>

#include <boost/math/special_functions/binomial.hpp>

#include <cassert>
#include <iostream>
#include <ranges>
#include <vector>

#include "common.hpp"

/**
 * @brief Find N degree K Bernstein polonomials p_i(t) s.t.
 *
 * p_i(0) = 0
 * p_i(dt) = dx   for (dt, dx) in zip(dtrange, dxrange)
 * p_i^(d) (1) = p_{i+1}^(d) (0)  for d = 1, \ldots, D, i = 0, \ldots, N-2
 *
 * p_0^(k) = 0   for k = 1, \ldots, D
 * p_N^(k) = 0   for k = 1, \ldots, D
 *
 * \sum_i \int |p_i^(d) (t)|^2 dt   t : 0 -> dt is minimized
 */
template<int K, int D, std::ranges::range Rt, std::ranges::range Rx>
Eigen::Matrix<double, K + 1, 1> fit_polynomial_1d(const Rt & dtrange, const Rx & dxrange)
{
  const std::size_t N = std::min(std::ranges::size(dtrange), std::ranges::size(dxrange));

  // coefficient layout is
  //   [ a00 a01 ... a0K a10 a11 ... a1K   ...   a{N-1}0 a{N-1}1 ... a{N-1}K ]
  // where p_i(t) = ai0 + ai1 b1(t) + ... + aiK bk(t) defines p on [tvec(i), tvec(i+1)]

  const std::size_t N_coef = (K + 1) * N;

  // Constraints counting:
  //  - N interval beg constraints (set ai0 = 0 )
  //  - N interval end constraints (set aiK = dx)
  //  - D * (N - 1) derivative constraints
  //  - D curve beg constraints
  //  - D curve end constraints

  const std::size_t N_eq = 2 * N + D * (N - 1) + 2 * D;

  assert(N_coef >= N_eq);

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_eq, N_coef);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // counter for current inequality
  std::size_t M = 0;

  // add interval start and end constraints (2 * N)
  for (auto i = 0u; i < N; ++i) {
    const std::size_t pi_beg = i * (K + 1);
    A(M, pi_beg)             = 1;  //  ai0 = 0
    b(M++)                   = 0;

    A(M, pi_beg + K) = 1;  //  aiK = dx
    b(M++)           = dxrange[i];
  }

  // add curve start and end constraints
  for (auto d = 1u; d <= D; ++d) {
    double nfac_div_nmdfac = 1;
    for (int i = K - d; i <= K; ++i) { nfac_div_nmdfac *= i; }

    // start
    for (auto nu = 0u; nu <= K; ++nu) {
      // d:th derivative of b_nu_n(t) at 0 is zero
      if (nu <= d && d < K) {
        const double p_over_nu    = boost::math::binomial_coefficient<double>(d, nu);
        const double sign         = (nu + d) % 2 == 0 ? -1 : 1;
        const double b0_nu_n_pder = nfac_div_nmdfac * sign * p_over_nu;
        A(M, nu)                  = b0_nu_n_pder;
      }
    }
    b(M++) = 0;

    // d:th derivative of b_nu_n(t) at 1 is zero
    for (auto nu = 0u; nu <= K; ++nu) {
      if (K - nu <= d && d < K) {
        const double p_over_n_minus_nu = boost::math::binomial_coefficient<double>(d, K - nu);
        const double sign              = (K - nu) % 2 == 0 ? -1 : 1;
        const double b1_nu_n_pder      = nfac_div_nmdfac * sign * p_over_n_minus_nu;
        A(M, (K + 1) * (N - 1) + nu)   = b1_nu_n_pder;
      }
    }
    b(M++) = 0;
  }

  // add derivative constraints
  for (auto d = 1u; d <= D; ++d) {
    double nfac_div_nmdfac = 1;
    for (int i = K - d; i <= K; ++i) { nfac_div_nmdfac *= i; }

    for (auto i = 0u; i + 1 < N; ++i) {
      const auto j             = i + 1;
      const std::size_t pi_beg = i * (K + 1);
      const std::size_t pj_beg = j * (K + 1);

      for (auto nu = 0u; nu <= K; ++nu) {
        // d:th derivative of b_nu_n(t) at 1
        if (K - nu <= d && d < K) {
          const double p_over_n_minus_nu = boost::math::binomial_coefficient<double>(d, K - nu);
          const double sign              = (K - nu) % 2 == 0 ? -1 : 1;
          const double b1_nu_n_pder      = nfac_div_nmdfac * sign * p_over_n_minus_nu;
          A(M, pi_beg + nu)              = b1_nu_n_pder * dtrange[j];
        }

        // d:th derivative of b_nu_n(t) at 0
        if (nu <= d && d <= K) {
          const double p_over_nu    = boost::math::binomial_coefficient<double>(d, nu);
          const double sign         = (nu + d) % 2 == 0 ? -1 : 1;
          const double b0_nu_n_pder = nfac_div_nmdfac * sign * p_over_nu;

          A(M, pj_beg + nu) = -b0_nu_n_pder * dtrange[i];
        }
      }
      b(M++) = 0;
    }
  }

  // coefficient constraint matrix is s.t. A x = b

  std::cout << A << std::endl;
  std::cout << "b: " << b.transpose() << std::endl;

  // need optimization matrix s.t. x' P x = \int p^(d) (t) dt
  //
  // it is equal to B int_0^1 B'

  Eigen::Matrix<double, K + 1, K + 1> IntMat = Eigen::Matrix<double, K + 1, K + 1>::Zero();
  for (auto i = 0u; i <= K; ++i) {
    for (auto j = 0u; j <= K; ++j) {
      if (i >= D && j >= D) {
        if (i + j == 2 * D) {
          IntMat(i, j) = 1;
        } else {
          IntMat(i, j) = static_cast<double>((i - D + 1) * (j - D + 1)) / (i + j - 2 * D);
        }
      }
    }
  }

  std::cout << "IntMat" << std::endl << IntMat << std::endl;

  return Eigen::Matrix<double, K + 1, 1>::Zero();
}
