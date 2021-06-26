#ifndef SMOOTH__NLS_HPP_
#define SMOOTH__NLS_HPP_

#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include <iostream>
#include <numeric>

#include "concepts.hpp"
#include "diff.hpp"
#include "internal/utils.hpp"
#include "internal/lmpar.hpp"

namespace smooth {

struct NlsOptions
{
  /// relative parameter tolerance for convergence
  double ptol{1e-6};
  /// relative function tolerance for convergence
  double ftol{1e-6};
  /// maximum number of iterations
  std::size_t max_iter{1000};
  /// solver verbosity level
  int verbosity{0};
};

/**
 * @brief Find a minimum of the non-linear least-squares problem
 *
 *  \min_{wrt...} \sum_i \| f(wrt...)_i \|^2
 *
 * @param f residuals to minimize
 * @param wrt arguments to f
 *
 * TODO add optional verbosity, termination conditions, parameters, output
 */
template<diff::Type difftype, typename _F, typename _Wrt>
void minimize(_F && f, _Wrt && x, const NlsOptions & opts = NlsOptions{})
{
  // evaluate residuals and jacobian at initial point
  auto [r, J] = diff::dr<difftype>(f, x);

  // extract some properties from jacobian
  static constexpr bool is_sparse =
    std::is_base_of_v<Eigen::SparseMatrixBase<decltype(J)>, decltype(J)>;
  static constexpr int Nx = decltype(J)::ColsAtCompileTime;
  const int nx            = J.cols();

  // scaling parameters
  Eigen::Matrix<double, Nx, 1> d(nx);
  if constexpr (is_sparse) {
    d = (Eigen::Matrix<double, 1, -1>::Ones(J.rows()) * J.cwiseProduct(J)).cwiseSqrt().transpose();
  } else {
    d = J.colwise().stableNorm().transpose();
  }

  // ensure scaling parameters are non-zero
  for (auto i = 0u; i != d.size(); ++i) {
    if (d[i] == 0) {
      d[i] = 1;
    }
  }

  double r_norm = r.stableNorm();
  double Delta  = 100. * d.stableNorm();  // TODO for Rn arguments we should multiply with norm(x)

  for (auto i = 0u; i != opts.max_iter; ++i) {
    // calculate step a via LM parameter algorithm
    const auto [lambda, a] = detail::lmpar(J, d, r, Delta);

    // evaluate function and jacobian at x + a
    auto x_plus_a = utils::tuple_plus(x, a, std::make_index_sequence<std::tuple_size_v<_Wrt>>{});
    const auto [r_cand, J_cand] = diff::dr<difftype>(f, x_plus_a);

    const double r_cand_norm = r_cand.stableNorm();
    const double Da_norm     = d.cwiseProduct(a).stableNorm();

    // calculate actual to predicted reduction
    const double act_red  = 1. - Eigen::numext::abs2(r_cand_norm / r_norm);
    const double fra2     = Eigen::numext::abs2((J * a).stableNorm() / r_norm);
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

    //// TAKE STEP IF SUCCESSFUL ////

    if (rho > 1e-4) {
      x      = x_plus_a;
      r      = r_cand;
      J      = J_cand;
      r_norm = r_cand_norm;

      // update scaling
      if constexpr (is_sparse) {
        d = d.cwiseMax((Eigen::Matrix<double, 1, -1>::Ones(J.rows()) * J.cwiseProduct(J))
                         .cwiseSqrt()
                         .transpose());
      } else {
        d = d.cwiseMax(J.colwise().stableNorm().transpose());
      }
    }

    //// PRINT STATUS ////
    // \todo Pretty-print solver steps
    if (opts.verbosity > 0)
    {
      std::cout << "Step " << i << ": " << r.sum() << std::endl;
    }

    //// CHECK FOR CONVERGENCE ////

    // function tolerance
    if (std::abs(act_red) < opts.ftol && pred_red < opts.ftol && rho <= 2.) {
      break;
    }

    // parameter tolerance
    // \todo a.size() should be norm(x) for non-angle states
    if (Da_norm < opts.ptol * a.size()) {
      break;
    }
  }
}

template<typename _F, typename _Wrt>
void minimize(_F && f, _Wrt && wrt, const NlsOptions & opts = NlsOptions{})
{
  minimize<diff::Type::DEFAULT>(std::forward<_F>(f), std::forward<_Wrt>(wrt), opts);
}

}  // namespace smooth

#endif  // SMOOTH__NLS_HPP_
