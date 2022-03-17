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

#ifndef SMOOTH__DIFF_HPP_
#define SMOOTH__DIFF_HPP_

/**
 * @file
 * @brief Differentiation on Manifolds.
 */

#include <Eigen/Core>
#include <type_traits>

#include "internal/utils.hpp"
#include "manifold.hpp"
#include "wrt.hpp"

namespace smooth {

// differentiation module
namespace diff {
namespace detail {

/**
 * @brief Numerical first-order differentiation in tangent space.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<std::size_t K = 1>
  requires(K >= 1 && K <= 2)
auto dr_numerical(auto && f, auto && x)
{
  using Wrt    = decltype(x);
  using Result = decltype(std::apply(f, x));
  using Scalar = ::smooth::Scalar<Result>;

  static constexpr auto NumArgs = std::tuple_size_v<std::decay_t<Wrt>>;

  static_assert(Manifold<Result>, "f(x) is not a Manifold");

  const Scalar eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());

  // arguments are modified below, so we create a copy of those that come in as const
  auto x_nc   = wrt_copy_if_const(std::forward<Wrt>(x));
  Result fval = std::apply(f, x_nc);

  // static sizes
  static constexpr Eigen::Index Nx = wrt_Dof<Wrt>();
  static constexpr Eigen::Index Ny = Dof<Result>;

  // dynamic sizes
  Eigen::Index nx = std::apply([](auto &&... args) { return (dof(args) + ...); }, x_nc);
  Eigen::Index ny = dof<Result>(fval);

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> J(ny, nx);

  if constexpr (K == 1) {
    Eigen::Index I0 = 0;
    utils::static_for<NumArgs>([&](auto i) {
      auto & w = std::get<i>(x_nc);
      using W  = std::decay_t<decltype(w)>;

      static constexpr Eigen::Index Nx_j = Dof<W>;
      const int nx_j                     = dof<W>(w);

      for (auto j = 0; j != nx_j; ++j) {
        Scalar eps_j = eps;
        if constexpr (std::is_base_of_v<Eigen::MatrixBase<W>, W>) {
          // scale step size if we are in Rn
          eps_j *= abs(w[j]);
          if (eps_j == 0.) { eps_j = eps; }
        }
        w             = rplus<W>(w, (eps_j * Eigen::Vector<Scalar, Nx_j>::Unit(nx_j, j)).eval());
        J.col(I0 + j) = rminus<Result>(std::apply(f, x_nc), fval) / eps_j;
        w             = rplus<W>(w, (-eps_j * Eigen::Vector<Scalar, Nx_j>::Unit(nx_j, j)).eval());
      }
      I0 += nx_j;
    });

    return std::make_pair(std::move(fval), std::move(J));
  }

  if constexpr (K == 2) {
    const auto sqrteps = std::sqrt(eps);

    Eigen::Matrix<Scalar, Nx, std::min(Nx, Ny) == -1 ? -1 : Nx * Ny> H(nx, nx * ny);

    Eigen::Index I0 = 0;
    utils::static_for<NumArgs>([&](auto i0) {
      auto & w0                           = std::get<i0>(x_nc);
      using W0                            = std::decay_t<decltype(w0)>;
      static constexpr Eigen::Index Nx_i0 = Dof<W0>;
      const int nx_i0                     = dof<W0>(w0);

      Eigen::Index I1 = 0;
      utils::static_for<NumArgs>([&](auto i1) {
        auto & w1                           = std::get<i1>(x_nc);
        using W1                            = std::decay_t<decltype(w1)>;
        static constexpr Eigen::Index Nx_i1 = Dof<W1>;
        const int nx_i1                     = dof<W1>(w1);

        for (auto k0 = 0; k0 != nx_i0; ++k0) {
          Scalar eps0 = sqrteps;
          if constexpr (std::is_base_of_v<Eigen::MatrixBase<W0>, W0>) {
            eps0 *= abs(w0[k0]);
            if (eps0 == 0.) { eps0 = sqrteps; }
          }

          w0               = rplus<W0>(w0, eps0 * Eigen::Vector<Scalar, Nx_i0>::Unit(nx_i0, k0));
          const Result F10 = std::apply(f, x_nc);
          w0               = rplus<W0>(w0, -eps0 * Eigen::Vector<Scalar, Nx_i0>::Unit(nx_i0, k0));

          const Eigen::Matrix<Scalar, Ny, 1> d1 = rminus(F10, fval);

          J.col(I0 + k0) = d1 / eps0;

          for (auto k1 = 0; k1 < nx_i1; ++k1) {
            Scalar eps1 = sqrteps;
            if constexpr (std::is_base_of_v<Eigen::MatrixBase<W1>, W1>) {
              eps1 *= abs(w1[k1]);
              if (eps1 == 0.) { eps1 = sqrteps; }
            }

            // do this in order to ensure we return to same point on spaces with non-zero brackets
            w1               = rplus<W1>(w1, eps1 * Eigen::Vector<Scalar, Nx_i1>::Unit(nx_i1, k1));
            const Result F01 = std::apply(f, x_nc);
            w0               = rplus<W0>(w0, eps0 * Eigen::Vector<Scalar, Nx_i0>::Unit(nx_i0, k0));
            const Result F11 = std::apply(f, x_nc);
            w0               = rplus<W0>(w0, -eps0 * Eigen::Vector<Scalar, Nx_i0>::Unit(nx_i0, k0));
            w1               = rplus<W1>(w1, -eps1 * Eigen::Vector<Scalar, Nx_i1>::Unit(nx_i1, k1));

            const Eigen::Matrix<Scalar, Ny, 1> d2 = (rminus(F11, F01) - d1) / eps0 / eps1;
            for (auto j = 0u; j < ny; ++j) { H(I0 + k0, j * nx + I1 + k1) = d2(j); }
          }
        }
        I1 += nx_i1;
      });
      I0 += nx_i0;
    });

    return std::make_tuple(std::move(fval), std::move(J), std::move(H));
  }
}

/// @brief Callable types that provide first-order derivative
template<class F, class Wrt>
concept diffable_order1 = requires(F && f, Wrt && wrt)
{
  {std::apply(f, wrt)};
  {std::apply(std::bind_front(std::mem_fn(&std::decay_t<decltype(f)>::jacobian), f), wrt)};
};

/// @brief Callable types that provide second-order derivative
template<class F, class Wrt>
concept diffable_order2 = diffable_order1<F, Wrt> && requires(F && f, Wrt && wrt)
{
  {std::apply(std::bind_front(std::mem_fn(&std::decay_t<decltype(f)>::hessian), f), wrt)};
};

}  // namespace detail

/**
 * @brief Available differentiation methods
 */
enum class Type {
  Numerical,  ///< Numerical (forward) derivatives
  Autodiff,   ///< Uses the autodiff (https://autodiff.github.io) library; requires  \p
              ///< compat/autodiff.hpp
  Ceres,      ///< Uses the Ceres (http://ceres-solver.org) built-in autodiff; requires \p
              ///< compat/ceres.hpp
  Analytic,   ///< Hand-coded derivative, @see diffable_order1, diffable_order2
  Default     ///< Select based on availability (Analytic > Autodiff > Ceres > Numerical)
};

static constexpr Type DefaultType =
#ifdef SMOOTH_DIFF_AUTODIFF
  Type::Autodiff;
#elif defined SMOOTH_DIFF_CERES
  Type::Ceres;
#else
  Type::Numerical;
#endif

/**
 * @brief Differentiation in tangent space
 *
 * @tparam K differentiation order (0, 1 or 2)
 * @tparam D differentiation method to use
 *
 * First derivatives are returned as a matrix df s.t.
 * df(i, j) = dfi / dxj, where fi is the i:th degree of freedom of f, and xj the j:th degree of
 * freedom of x.
 *
 * Second derivatives are stored as a horizontally stacked block matrix
 * d2f = [ d2f0 d2f1 ... d2fN ],
 * where d2fi(j, k) = d2fi / dxjxk is the Hessian of the i:th degree of freedom of f.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return {f(x)} for K = 0, {f(x), dr f(x)} for K = 1, {f(x), dr f(x), d2r f(x)} for K = 2
 *
 * @note Only scalar functions suppored for K = 2
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<std::size_t K, Type D>
auto dr(auto && f, auto && x)
{
  using F   = decltype(f);
  using Wrt = decltype(x);

  if constexpr (K == 0u) {
    // Only function value needed

    return std::make_tuple(std::apply(f, x));

  } else if constexpr (D == Type::Numerical) {
    // Numerical

    return detail::dr_numerical<K>(std::forward<F>(f), std::forward<Wrt>(x));

  } else if constexpr (D == Type::Autodiff) {
    // Autodiff

#ifdef SMOOTH_DIFF_AUTODIFF
    return dr_autodiff<K>(std::forward<F>(f), std::forward<Wrt>(x));
#else
    static_assert(D != Type::Autodiff, "compat/autodiff.hpp must be included before diff.hpp");
#endif

  } else if constexpr (D == Type::Ceres) {
    // Ceres

#ifdef SMOOTH_DIFF_CERES
    static_assert(K == 1, "Only K = 1 supported with Ceres");
    return dr_ceres(std::forward<F>(f), std::forward<Wrt>(x));
#else
    static_assert(D != Type::Ceres, "compat/ceres.hpp header must be included before diff.hpp");
#endif

  } else if constexpr (D == Type::Analytic) {
    // Analytic

    if constexpr (K == 1) {
      return std::make_tuple(
        std::apply(f, x),
        std::apply(
          [&f](auto &&... args) -> decltype(auto) {
            return f.jacobian(std::forward<decltype(args)>(args)...);
          },
          x));
    } else if constexpr (K == 2) {
      return std::make_tuple(
        std::apply(f, x),
        std::apply(
          [&f](auto &&... args) -> decltype(auto) {
            return f.jacobian(std::forward<decltype(args)>(args)...);
          },
          x),
        std::apply(
          [&f](auto &&... args) -> decltype(auto) {
            return f.hessian(std::forward<decltype(args)>(args)...);
          },
          x));
    }

  } else if constexpr (D == Type::Default) {
    // Default

    // If analytical derivatives exist
    if constexpr (K == 1 && detail::diffable_order1<F, Wrt>) {
      return dr<K, Type::Analytic>(std::forward<F>(f), std::forward<Wrt>(x));
    } else if constexpr (K == 2 && detail::diffable_order2<F, Wrt>) {
      return dr<K, Type::Analytic>(std::forward<F>(f), std::forward<Wrt>(x));
    } else {
      // Use best available method
      return dr<K, DefaultType>(std::forward<F>(f), std::forward<Wrt>(x));
    }
  }
}

/**
 * @brief Differentiation in tangent space using default method
 *
 * @tparam K differentiation order
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<std::size_t K = 1>
auto dr(auto && f, auto && x)
{
  return dr<K, Type::Default>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x));
}

/**
 * @brief Derivative of matrix product.
 *
 * @param A matrix                                         [N x K]
 * @param dA derivative of A on horizontal Hessian form    [K x N*Nvar]
 * @param B matrix                                         [K x M]
 * @param dB derivative of B on horizontal Hessian form    [M x K*Nvar]
 * @return derivative of A * B on horizontal Hessian form
 */
template<typename At, typename dAt, typename Bt, typename dBt>
auto matrix_product(const At & A, const dAt & dA, const Bt & B, const dBt & dB)
{
  using Scalar = std::common_type_t<
    typename At::Scalar,
    typename dAt::Scalar,
    typename Bt::Scalar,
    typename dBt::Scalar>;

  static constexpr int N    = At::ColsAtCompileTime;
  static constexpr int M    = Bt::RowsAtCompileTime;
  static constexpr int Nvar = []() -> int {
    if constexpr (dAt::ColsAtCompileTime > 0 && N > 0) {
      return dAt::ColsAtCompileTime / N;
    } else if (dBt::ColsAtCompileTime > 0 && M > 0) {
      return dBt::ColsAtCompileTime / M;
    } else {
      return -1;
    }
  }();

  const auto n    = A.cols();
  const auto m    = B.rows();
  const auto nvar = dA.cols() / (n);

  assert(k == b.cols());
  assert(nvar == dB.size() / (m * k));

  static constexpr int dAB_cols = (M > 0 && Nvar > 0) ? M * Nvar : -1;

  Eigen::Matrix<Scalar, N, dAB_cols> dAB = B.transpose() * dA;
  for (auto i = 0u; i < n; ++i) {
    for (auto j = 0u; j < m; ++j) {
      dAB.template middleCols<Nvar>(i * nvar, nvar) +=
        A(i, j) * dB.template middleCols<Nvar>(j * Nvar, nvar);
    }
  }
  return dAB;
}

/**
 * @brief (Right) Hessian of composed function \f$ (f \circ g)(x) \f$.
 *
 * @param Jf (Right) Jacobian of f at y = g(x)  [No x Ny   ]
 * @param Hf (Right) Hessian of f at y = g(x)   [Ny x No*Ny]
 * @param Jg (Right) Jacobian of g at x         [Ny x Nx   ]
 * @param Hg (Right) Hessian of g at x          [Nx x Ny*Nx]
 *
 * @return Hessian of size [No x No*Nx]
 */
template<typename JfT, typename HfT, typename JgT, typename HgT>
auto d2r_fog(const JfT & Jf, const HfT & Hf, const JgT & Jg, const HgT & Hg)
{
  using Scalar = std::common_type_t<
    typename JfT::Scalar,
    typename HfT::Scalar,
    typename JgT::Scalar,
    typename HgT::Scalar>;

  static constexpr int No = JfT::RowsAtCompileTime;
  static constexpr int Ny = JfT::ColsAtCompileTime;
  static constexpr int Nx = JgT::ColsAtCompileTime;

  const auto no = Jf.rows();
  const auto ny = Jf.cols();

  [[maybe_unused]] const auto ni = Jg.rows();
  const auto nx                  = Jg.cols();

  // check some dimensions
  assert(ny == ni);
  assert(Hf.rows() == ny);
  assert(Hf.cols() == no * ny);
  assert(Hg.rows() == nx);
  assert(Hg.cols() == ni * nx);

  Eigen::Matrix<Scalar, No, (No == -1 || Nx == -1) ? -1 : No * Nx> ret(no, no * nx);
  ret.setZero();

  for (auto i = 0u; i < no; ++i) {
    ret.template block<Nx, Nx>(0, i * nx, nx, nx) +=
      Jg.transpose() * Hf.template middleCols<Ny>(i * ny, ny) * Jg;
  }

  for (auto i = 0u; i < Jf.outerSize(); ++i) {
    for (Eigen::InnerIterator it(Jf, i); it; ++it) {
      ret.template block<Nx, Nx>(0, it.row() * nx) +=
        it.value() * Hg.template middleCols<Nx>(it.col() * nx, nx);
    }
  }

  return ret;
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__DIFF_HPP_
