// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__OPTIM_HPP_
#define SMOOTH__OPTIM_HPP_

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

#include "diff.hpp"
#include "internal/lmpar.hpp"
#include "internal/lmpar_sparse.hpp"
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
  auto [r, J] = diff::dr<D>(f, x);

  // extract some properties from jacobian
  static constexpr bool is_sparse =
    std::is_base_of_v<Eigen::SparseMatrixBase<decltype(J)>, decltype(J)>;
  static constexpr int Nx = decltype(J)::ColsAtCompileTime;
  const int nx            = J.cols();

  // scaling parameters
  Eigen::Vector<double, Nx> d(nx);
  if constexpr (is_sparse) {
    d = (Eigen::RowVectorXd::Ones(J.rows()) * J.cwiseProduct(J)).cwiseSqrt().transpose();
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
    using std::cout, std::left, std::endl, std::setw, std::right;
    // clang-format off
    cout << "================= Levenberg-Marquardt Solver ================" << endl;
    cout << "Solving NLSQ with n=" << J.cols() << ", m=" << J.rows() << endl;
    cout << setw(8)  << right << "ITER"
         << setw(14) << right << "|| r ||"
         << setw(14) << right << "RED"
         << setw(14) << right << "|| D * a ||"
         << setw(10) << right << "TIME" << std::endl;
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
    const auto [r_cand, J_cand] = diff::dr<D>(f, x_plus_a);

    const double r_cand_norm = r_cand.stableNorm();
    const double Da_norm     = d.cwiseProduct(a).stableNorm();

    // calculate actual to predicted reduction
    const Eigen::Vector<double, decltype(J)::RowsAtCompileTime> Ja = J * a;
    const double act_red  = 1. - Eigen::numext::abs2(r_cand_norm / r_norm);
    const double fra2     = Eigen::numext::abs2(Ja.stableNorm() / r_norm);
    const double fra3     = Eigen::numext::abs2(std::sqrt(lambda) * Da_norm / r_norm);
    const double pred_red = fra2 + 2. * fra3;
    const double rho      = act_red / pred_red;

    // update trust region following Moré (1978)
    if (rho < 0.25) {
      double mu;
      if (r_cand_norm <= r_norm) {
        mu = 0.5;
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
      using std::cout, std::setw, std::right, std::endl;
      // clang-format off
      cout << setw(7) << right << iter << ":"
           << std::scientific
           << setw(14) << right << r_cand_norm
           << setw(14) << right << act_red
           << setw(14) << right << Da_norm
           << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()
           << endl;
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
        d = d.cwiseMax(
          (Eigen::RowVectorXd::Ones(J.rows()) * J.cwiseProduct(J)).cwiseSqrt().transpose());
      } else {
        d = d.cwiseMax(J.colwise().stableNorm().transpose());
      }
    }

    //// CHECK FOR CONVERGENCE ////

    // function tolerance
    if (std::abs(act_red) < opts.ftol && pred_red < opts.ftol && rho <= 2.) { break; }

    // parameter tolerance
    // TODO a.size() should be norm(x) for non-angle states
    if (Da_norm < opts.ptol * a.size()) { break; }
  }

  //// PRINT STATUS ////
  if (opts.verbose) {
    using std::cout, std::left, std::right, std::setw, std::endl;

    // clang-format off
    cout << "NLSQ solver summary:" << endl;
    cout << setw(25) << left << "Iterations"                << setw(10) << right << iter                                                                                    << endl;
    cout << setw(25) << left << "Total time (microseconds)" << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()     << endl;
    cout << "=============================================================" << endl;
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
  minimize<diff::Type::DEFAULT>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x), opts);
}

}  // namespace smooth

#endif  // SMOOTH__OPTIM_HPP_
