// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "../diff.hpp"

namespace smooth::diff {

namespace detail {

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
  static constexpr auto Nx = wrt_Dof<Wrt>();
  static constexpr auto Ny = Dof<Result>;

  // dynamic sizes
  const auto nx = std::apply([](auto &&... args) { return (dof(args) + ...); }, x_nc);
  const auto ny = dof<Result>(fval);

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> J(ny, nx);

  if constexpr (K == 1) {
    Eigen::Index I0 = 0;
    utils::static_for<NumArgs>([&](auto i) {
      auto & w = std::get<i>(x_nc);
      using W  = std::decay_t<decltype(w)>;

      static constexpr auto Nx_j = Dof<W>;
      const auto nx_j            = dof<W>(w);

      for (auto j = 0; j != nx_j; ++j) {
        Scalar eps_j = eps;
        if constexpr (std::is_base_of_v<Eigen::MatrixBase<W>, W>) {
          // scale step size if we are in Rn
          eps_j *= abs(w[j]);
          if (eps_j == Scalar(0.)) { eps_j = eps; }
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
      auto & w0                   = std::get<i0>(x_nc);
      using W0                    = std::decay_t<decltype(w0)>;
      static constexpr auto Nx_i0 = Dof<W0>;
      const auto nx_i0            = dof<W0>(w0);

      Eigen::Index I1 = 0;
      utils::static_for<NumArgs>([&](auto i1) {
        auto & w1                   = std::get<i1>(x_nc);
        using W1                    = std::decay_t<decltype(w1)>;
        static constexpr auto Nx_i1 = Dof<W1>;
        const auto nx_i1            = dof<W1>(w1);

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
          [&f](auto &&... args) -> decltype(auto) { return f.jacobian(std::forward<decltype(args)>(args)...); }, x));
    } else if constexpr (K == 2) {
      return std::make_tuple(
        std::apply(f, x),
        std::apply(
          [&f](auto &&... args) -> decltype(auto) { return f.jacobian(std::forward<decltype(args)>(args)...); }, x),
        std::apply(
          [&f](auto &&... args) -> decltype(auto) { return f.hessian(std::forward<decltype(args)>(args)...); }, x));
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
      static constexpr Type DefaultType =
#ifdef SMOOTH_DIFF_AUTODIFF
        Type::Autodiff;
#elif defined SMOOTH_DIFF_CERES
        Type::Ceres;
#else
        Type::Numerical;
#endif
      return dr<K, DefaultType>(std::forward<F>(f), std::forward<Wrt>(x));
    }
  }
}

template<std::size_t K, Type D, std::size_t... Idx>
auto dr(auto && f, auto && x, std::index_sequence<Idx...>)
{
  // function taking reduced argument
  auto f_wrapped = [f = std::forward<decltype(f)>(f), &x](auto &&... arg_red) {
    // cast x to scalar type of arg_red
    using ArgScalar = std::common_type_t<Scalar<std::decay_t<decltype(arg_red)>>...>;
    auto arg_full   = smooth::wrt_cast<ArgScalar>(x);

    // copy in arg_red to correct positions in x
    ((std::get<Idx>(arg_full) = arg_red), ...);

    return std::apply(f, arg_full);
  };

  auto x_red = std::forward_as_tuple(std::get<Idx>(x)...);
  return dr<K, D>(std::move(f_wrapped), std::move(x_red));
}

template<std::size_t K>
auto dr(auto && f, auto && x)
{
  return dr<K, Type::Default>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x));
}

template<std::size_t K, std::size_t... Idx>
auto dr(auto && f, auto && x, std::index_sequence<Idx...> idx)
{
  return dr<K, Type::Default>(std::forward<decltype(f)>(f), std::forward<decltype(x)>(x), idx);
}

}  // namespace smooth::diff
