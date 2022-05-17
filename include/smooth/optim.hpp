// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Non-linear least squares optimization on Manifolds.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "detail/lmpar.hpp"
#include "detail/lmpar_sparse.hpp"
#include "diff.hpp"
#include "manifold.hpp"
#include "wrt.hpp"

using std::chrono::duration_cast, std::chrono::microseconds;

namespace smooth {

/**
 * @brief Optimization options.
 */
struct MinimizeOptions
{
  /// relative parameter tolerance for convergence
  double ptol{1e-6};
  /// relative function tolerance for convergence
  double ftol{1e-6};
  /// maximum number of iterations
  std::size_t max_iter{1000};
  /// print solver status to stdout
  bool verbose{0};
};

/**
 * @brief Find a minimum of the non-linear least-squares problem
 *
 * \f[
 *  \min_{x} \sum_i \| f(x)_i \|^2
 * \f]
 *
 * @tparam D differentiation method to use in solver (see diff::Type in diff.hpp)
 * @param f residuals to minimize
 * @param x reference tuple of arguments to f
 * @param opts solver options
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<diff::Type D>
void minimize(auto && f, auto && x, const MinimizeOptions & opts = MinimizeOptions{})
{
  // evaluate residuals and jacobian at initial point
  auto [r, J] = diff::dr<1, D>(f, x);

  using JType = std::decay_t<decltype(J)>;

  // extract some properties from jacobian
  static constexpr bool is_sparse = std::is_base_of_v<Eigen::SparseMatrixBase<JType>, JType>;
  static constexpr auto Nx        = JType::ColsAtCompileTime;
  const auto nx                   = J.cols();

  // scaling parameters
  Eigen::Vector<double, Nx> d(nx);
  if constexpr (is_sparse) {
    d = (Eigen::RowVector<double, JType::RowsAtCompileTime>::Ones(J.rows()) * J.cwiseProduct(J))
          .cwiseSqrt()
          .transpose();
  } else {
    d = J.colwise().stableNorm().transpose();
  }

  // ensure scaling parameters are non-zero
  for (auto i = 0u; i != d.size(); ++i) {
    if (d[i] == 0) { d[i] = 1; }
  }

  double r_norm = r.stableNorm();
  double Delta  = 100. * d.stableNorm();  // TODO for Rn arguments we should multiply with norm(x)

  std::size_t iter = 0u;

  auto t0 = std::chrono::high_resolution_clock::now();

  if (opts.verbose) {
    using std::cout, std::left, std::setw, std::right;
    // clang-format off
    cout << "================= Levenberg-Marquardt Solver ================\n";
    cout << "Solving NLSQ with n=" << J.cols() << ", m=" << J.rows() << '\n';
    cout << setw(8)  << right << "ITER"
         << setw(14) << right << "|| r ||"
         << setw(14) << right << "RED"
         << setw(14) << right << "|| D * a ||"
         << setw(10) << right << "TIME" << '\n';
    // clang-format on
  }

  for (; iter < opts.max_iter; ++iter) {
    // calculate step a via LM parameter algorithm
    Eigen::Vector<double, Nx> a(nx);
    double lambda;
    if constexpr (is_sparse) {
      std::tie(lambda, a) = detail::lmpar_sparse(J, d, r, Delta);
    } else {
      std::tie(lambda, a) = detail::lmpar(J, d, r, Delta);
    }

    // evaluate function and jacobian at x + a
    const auto x_plus_a         = wrt_rplus(x, a);
    const auto [r_cand, J_cand] = diff::dr<1, D>(f, x_plus_a);

    const double r_cand_norm = r_cand.stableNorm();
    const double Da_norm     = d.cwiseProduct(a).stableNorm();

    // calculate actual to predicted reduction
    const Eigen::Vector<double, JType::RowsAtCompileTime> Ja = J * a;
    const double act_red  = 1. - Eigen::numext::abs2(r_cand_norm / r_norm);
    const double fra2     = Eigen::numext::abs2(Ja.stableNorm() / r_norm);
    const double fra3     = Eigen::numext::abs2(std::sqrt(lambda) * Da_norm / r_norm);
    const double pred_red = fra2 + 2. * fra3;
    const double rho      = act_red / pred_red;

    // update trust region following Moré (1978)
    if (rho < 0.25) {
      double mu = 0.5;
      if (r_cand_norm <= r_norm) {
        // leave at 0.5
      } else if (r_cand_norm <= 10 * r_norm) {
        const double gamma = -fra2 - fra3;
        mu                 = std::clamp(gamma / (2. * gamma + act_red), 0.1, 0.5);
      } else {
        mu = 0.1;
      }
      Delta *= mu;
    } else if ((lambda == 0 && rho < 0.75) || rho > 0.75) {
      Delta = 2 * Da_norm;
    }

    if (opts.verbose) {
      using std::cout, std::setw, std::right;
      // clang-format off
      cout << setw(7) << right << iter << ":"
           << std::scientific
           << setw(14) << right << r_cand_norm
           << setw(14) << right << act_red
           << setw(14) << right << Da_norm
           << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()
           << '\n';
      // clang-format on
    }

    //// TAKE STEP IF SUCCESSFUL ////

    if (rho > 1e-4) {
      x      = x_plus_a;
      r      = r_cand;
      J      = J_cand;
      r_norm = r_cand_norm;

      // update scaling
      if constexpr (is_sparse) {
        d = (Eigen::RowVector<double, JType::RowsAtCompileTime>::Ones(J.rows()) * J.cwiseProduct(J))
              .cwiseSqrt()
              .transpose();
      } else {
        d = d.cwiseMax(J.colwise().stableNorm().transpose());
      }
    }

    //// CHECK FOR CONVERGENCE ////

    // function tolerance
    if (std::abs(act_red) < opts.ftol && pred_red < opts.ftol && rho <= 2.) { break; }

    // parameter tolerance
    // TODO a.size() should be norm(x) for non-angle states
    if (Da_norm < opts.ptol * static_cast<double>(a.size())) { break; }
  }

  //// PRINT STATUS ////
  if (opts.verbose) {
    using std::cout, std::left, std::right, std::setw;

    // clang-format off
    cout << "NLSQ solver summary:\n";
    cout << setw(25) << left << "Iterations"                << setw(10) << right << iter                                                                                    << '\n';
    cout << setw(25) << left << "Total time (microseconds)" << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()     << '\n';
    cout << "=============================================================\n";
    // clang-format on
  }
}

/**
 * @brief Find a minimum of the non-linear least-squares problem
 *
 * \f[
 *  \min_{x} \sum_i \| f(x)_i \|^2
 * \f]
 *
 * @param f residuals to minimize
 * @param x reference tuple of arguments to f
 * @param opts solver options
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
void minimize(auto && f, auto && x, const MinimizeOptions & opts = MinimizeOptions{})
{
  minimize<diff::Type::Default>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x), opts);
}

}  // namespace smooth
