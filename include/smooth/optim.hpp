// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <chrono>
#include <memory>
#include <utility>

#include <Eigen/Sparse>

#ifdef SMOOTH_HAS_FMT
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/format.h>
#else
#include <iostream>
#endif

#include "detail/math.hpp"
#include "diff.hpp"
#include "optim/tr_solver.hpp"
#include "optim/tr_strategy.hpp"

SMOOTH_BEGIN_NAMESPACE

struct MinimizeOptions
{
  /// strategy
  std::shared_ptr<TrustRegionStrategy> strat{std::make_shared<CeresStrategy>()};
  /// relative parameter tolerance for convergence
  double ptol{1e-6};
  /// relative function tolerance for convergence
  double ftol{1e-6};
  /// maximum number of iterations
  std::size_t max_iter{1000};
  /// print solver status to stdout
  bool verbose{false};
};

struct SolveResult
{
  enum class Status { Ftol, Ptol, MaxIters } status;
  unsigned iter;
  std::chrono::nanoseconds time;
};

/**
 * @brief Find a minimum of the non-linear least-squares problem
 *
 * \f[
 *  \min_{x} \sum_i \| f(x)_i \|^2
 * \f]
 *
 * @tparam D differentiation method to use in solver (see diff::Type in diff.hpp)
 * @param f function to minimize
 * @param x reference tuple of arguments to f
 * @param cb callback cb: decltype(x) -> void to be called on each iteration
 * @param opts solver options
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<diff::Type D>
SolveResult minimize(auto && f, auto && x, auto && cb, const MinimizeOptions & opts = {})
  requires(!std::is_same_v<std::decay_t<decltype(cb)>, MinimizeOptions>)
{
  std::optional<SolveResult::Status> status = {};
  const auto t0                             = std::chrono::high_resolution_clock::now();
  auto iter                                 = 0u;

  // execute callback on initial value
  std::apply(cb, x);

  for (; iter < opts.max_iter && !status.has_value(); ++iter) {
    // evaluate residuals and jacobian
    const auto [r, J] = diff::dr<1, D>(f, x);

    using JType             = std::decay_t<decltype(J)>;
    static constexpr auto N = JType::ColsAtCompileTime;

    if (opts.verbose && iter == 0) {
#ifdef SMOOTH_HAS_FMT
      fmt::print("{0:x^69s}\n{1:x^69s}\n{0:x^69s}\n", "", "   NLS SOLVER   ");
      fmt::print("Optimizing {} residuals in {} degrees of freedom\n", J.rows(), J.cols());
#else
      std::cout << "#define SMOOTH_HAS_FMT and link with fmt to see minimizer progress.\n";
#endif
    }

    // diagonal scaling parameters
    static constexpr auto clamper    = [](double el) { return std::clamp(el, 1e-6, 1e32); };
    const Eigen::Vector<double, N> d = colwise_norm(J).unaryExpr(clamper);

    // trust region step
    const double Delta      = opts.strat->get_delta();
    const auto [dx, lambda] = solve_trust_region(J, d, r, Delta);
    const auto xp           = wrt_rplus(x, dx);

    // actual to relative reduction
    const double r_n      = r.stableNorm();
    const double actu_red = 1. - fpow<2>(std::apply(f, xp).stableNorm() / r_n);
    const double pred_red = 1. - fpow<2>((r + J * dx).stableNorm() / r_n);
    const double rho      = actu_red / pred_red;

    // update trust region
    const bool take_step = opts.strat->step_and_update(rho);

    if (opts.verbose) {
#ifdef SMOOTH_HAS_FMT
      using namespace fmt;          // NOLINT
      using namespace std::chrono;  // NOLINT
      if (iter % 25 == 0) {
        print("{:<6s}", "ITER");
        print("{:^10s}", "TIME");
        print(emphasis::bold, "{:^12s}", "∥r∥");
        print("{:^9s}", "Δ");
        print("{:^9}", "ρ");
        print("{:^9s}", "∥D dx∥");
        print("{:^9s}", "∥D∥");
        print("{:^6s}", "STEP");
        print("\n");
      }
      const auto step_col = take_step ? color::dark_green : color::dark_red;
      const auto rho_col  = rho > 0 ? color::dark_green : color::dark_red;

      print("{:<6}", iter);
      print("{:>9} ", duration_cast<milliseconds>(high_resolution_clock::now() - t0));
      print(emphasis::bold, "{:^12.4e}", r_n);
      print("{:^9.1e}", Delta);
      print(fg(rho_col), "{:^9.1e}", rho);
      print("{:^9.1e}", d.cwiseProduct(dx).stableNorm());
      print("{:^9.1e}", d.norm());
      print(fg(step_col), "{:^5s}", take_step ? "  ✔️ " : " ❌ ");
      print("\n");
#endif
    }

    // step
    if (r_n == 0 || pred_red <= 0 || take_step) {
      x = xp;

      // execute callback on updated value
      std::apply(cb, x);

      // check for convergence
      if (std::abs(actu_red) < opts.ftol && pred_red < opts.ftol && rho <= 2.) {
        status = SolveResult::Status::Ftol;
      } else if (d.cwiseProduct(dx).stableNorm() < opts.ptol * static_cast<double>(dx.size())) {
        status = SolveResult::Status::Ptol;
      }
    }
  }

  if (opts.verbose) {
#ifdef SMOOTH_HAS_FMT
    using namespace fmt;          // NOLINT
    using namespace std::chrono;  // NOLINT

    print("{0:x^69s}\n", "");
    print("{:>10s}: {}\n", "Total time", duration_cast<milliseconds>(high_resolution_clock::now() - t0));
    print("{:>10s}: {}\n", "Iterations", iter);
    print("{:>10s}: {:.2f}\n", "Objective", std::apply(f, x).norm());
    print("{0:x^69s}\n", "");
#endif
  }

  return {
    .status = status.value_or(SolveResult::Status::MaxIters),
    .iter   = iter,
    .time   = std::chrono::high_resolution_clock::now() - t0,
  };
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
template<diff::Type D>
SolveResult minimize(auto && f, auto && x, const MinimizeOptions & opts = {})
{
  static constexpr auto do_nothing = [](const auto &...) {};
  return minimize<D>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x), do_nothing, opts);
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
SolveResult minimize(auto && f, auto && x, const MinimizeOptions & opts = {})
{
  static constexpr auto do_nothing = [](const auto &...) {};
  return minimize<diff::Type::Default>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x), do_nothing, opts);
}

SMOOTH_END_NAMESPACE
