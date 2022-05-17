// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include "../lie_group_sparse.hpp"

namespace smooth {

/// @brief Forward-declare
// clang-format off
template<typename D> class C1Base;
template<typename D> class SE2Base;
template<typename D> class SE3Base;
template<typename D> class SO2Base;
template<typename D> class BundleBase;
// clang-format on

template<LieGroup G>
void ad_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a)
{
  assert(sp.isCompressed());
  sp.coeffs().setZero();
  for (auto k = 0u; k < Dof<G>; ++k) { sp += a(k) * generators_sparse<G>[k]; }
  assert(sp.isCompressed());
}

template<LieGroup G, bool Inv>
void dr_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0)
{
  using T = traits::lie_sparse<G>;
  assert(sp.isCompressed());
  assert(sp.rows() >= i0 + Dof<G>);
  assert(sp.cols() >= i0 + Dof<G>);
  if constexpr (IsCommutative<G>) {
    // identity matrix
    for (auto i = 0u; i < a.size(); ++i) { sp.coeffRef(i0 + i, i0 + i) = 1; }
  } else if constexpr (!Inv && requires { T::dr_exp_sparse(sp, a, i0); }) {
    // use specialized method if one exists
    T::dr_exp_sparse(sp, a, i0);
  } else if constexpr (Inv && requires { T::dr_expinv_sparse(sp, a, i0); }) {
    // use specialized method if one exists
    T::dr_expinv_sparse(sp, a, i0);
  } else {
    // fall back on dense method+pattern
    const TangentMap<G> D = [&] {
      if constexpr (Inv)
        return dr_expinv<G>(a);
      else
        return dr_exp<G>(a);
    }();
    for (auto i = 0u; i < Dof<G>; ++i) {
      for (Eigen::InnerIterator it(d_exp_sparse_pattern<G>, i); it; ++it) {
        sp.coeffRef(i0 + it.row(), i0 + it.col()) = D(it.row(), it.col());
      }
    }
  }
  assert(sp.isCompressed());
}

template<LieGroup G>
inline void
dr_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0)
{
  dr_exp_sparse<G, true>(sp, a, i0);
}

template<LieGroup G, bool Inv>
inline void
d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0)
{
  using T = traits::lie_sparse<G>;
  assert(sp.isCompressed());
  assert(sp.rows() >= i0 + Dof<G>);
  assert(sp.cols() >= sp.rows() * (i0 + Dof<G>));
  if constexpr (IsCommutative<G>) {
    // zero matrix--do nothing
  } else if constexpr (!Inv && requires { T::d2r_exp_sparse(sp, a, i0); }) {
    // use specialized method if one exists
    T::d2r_exp_sparse(sp, a, i0);
  } else if constexpr (Inv && requires { T::d2r_expinv_sparse(sp, a, i0); }) {
    // use specialized method if one exists
    T::d2r_expinv_sparse(sp, a, i0);
  } else {
    // fall back on dense method + pattern
    const Hessian<G> D = [&] {
      if constexpr (Inv)
        return d2r_expinv<G>(a);
      else
        return d2r_exp<G>(a);
    }();
    for (auto i = 0u; i < Dof<G> * Dof<G>; ++i) {
      for (Eigen::InnerIterator it(d2_exp_sparse_pattern<G>, i); it; ++it) {
        const auto block = i0 + (it.col() / Dof<G>);
        const auto row   = i0 + it.row();
        const auto col   = i0 + (it.col() % Dof<G>);

        sp.coeffRef(row, sp.rows() * block + col) = D(it.row(), it.col());
      }
    }
  }
  assert(sp.isCompressed());
}

template<LieGroup G>
inline void
d2r_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0)
{
  return d2r_exp_sparse<G, true>(sp, a, i0);
}

//////////////////////////////////////////////////////
//// SPARSITY INFORMATION FOR SPECIFIC LIE GROUPS ////
//////////////////////////////////////////////////////

namespace traits {
/// @brief Sparsity info for SE2 types.
template<typename G>
  requires(std::is_base_of_v<SE2Base<G>, G>)
struct lie_sparse<G>
{
  static inline Eigen::SparseMatrix<Scalar<G>> d_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(3, 3);
    for (auto i = 0u; i < 2; ++i) {
      for (auto j = 0u; j < 3; ++j) { ret.insert(i, j) = Scalar<G>(0); }
    }
    ret.insert(2, 2) = Scalar<G>(0);
    ret.makeCompressed();
    return ret;
  }();

  static inline Eigen::SparseMatrix<Scalar<G>> d2_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(3, 9);
    ret.insert(2, 0) = 0;
    ret.insert(2, 1) = 0;
    ret.insert(2, 2) = 0;
    ret.insert(0, 2) = 0;
    ret.insert(1, 2) = 0;

    ret.insert(2, 3) = 0;
    ret.insert(2, 4) = 0;
    ret.insert(2, 5) = 0;
    ret.insert(0, 5) = 0;
    ret.insert(1, 5) = 0;
    ret.makeCompressed();
    return ret;
  }();
};

/// @brief Sparsity info for SE3 types.
template<typename G>
  requires(std::is_base_of_v<SE3Base<G>, G>)
struct lie_sparse<G>
{
  static inline Eigen::SparseMatrix<Scalar<G>> d_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(6, 6);
    for (auto i = 0u; i < 6; ++i) {
      for (auto j = 0u; j < 6; ++j) {
        if (i < 3 || j >= 3) { ret.insert(i, j) = (i == j) ? Scalar<G>(1) : Scalar<G>(0); }
      }
    }
    ret.makeCompressed();
    return ret;
  }();

  static inline Eigen::SparseMatrix<Scalar<G>> d2_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(6, 36);
    for (auto i = 3u; i < 6u; ++i) {
      for (auto j = 0u; j < 18u; ++j) { ret.insert(i, j) = Scalar<G>(0); }
    }
    for (auto k = 0u; k < 3u; ++k) {
      for (auto i = 0u; i < 3u; ++i) {
        for (auto j = 0u; j < 3u; ++j) {
          ret.insert(i, 3 + 6 * k + j)          = Scalar<G>(0);
          ret.insert(3 + i, 18 + 3 + 6 * k + j) = Scalar<G>(0);
        }
      }
    }
    ret.makeCompressed();
    return ret;
  }();
};

/// @brief Sparsity info for Bundle types.
template<typename G>
  requires(std::is_base_of_v<BundleBase<G>, G>)
struct lie_sparse<G>
{
  /// @brief Sparsity pattern for dr_exp / dr_expinv
  static inline Eigen::SparseMatrix<Scalar<G>> d_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(Dof<G>, Dof<G>);
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      const auto & tmp           = ::smooth::d_exp_sparse_pattern<typename G::template PartType<I>>;
      for (auto i = 0u; i < tmp.outerSize(); ++i) {
        for (Eigen::InnerIterator it(tmp, i); it; ++it) {
          ret.insert(Dof0 + it.row(), Dof0 + it.col()) = it.value();
        }
      }
    });
    ret.makeCompressed();
    return ret;
  }();

  /// @brief Sparse calculation of dr_exp
  static void
  dr_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::dr_exp_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

  /// @brief Sparse calculation of dr_expinv
  static void
  dr_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::dr_expinv_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

  /// @brief Sparsity pattern for d2r_exp / d2r_expinv
  static inline Eigen::SparseMatrix<Scalar<G>> d2_exp_sparse_pattern = [] {
    Eigen::SparseMatrix<Scalar<G>> ret(Dof<G>, Dof<G> * Dof<G>);
    utils::static_for<G::BundleSize>([&ret](auto I) {
      const auto & tmp    = ::smooth::d2_exp_sparse_pattern<typename G::template PartType<I>>;
      constexpr auto Dof0 = G::template PartStart<I>;
      for (auto i = 0u; i < tmp.outerSize(); ++i) {
        for (Eigen::InnerIterator it(tmp, i); it; ++it) {
          const auto block = Dof0 + (it.col() / G::template PartDof<I>);
          const auto row   = it.row();
          const auto col   = Dof0 + (it.col() % G::template PartDof<I>);

          ret.insert(Dof0 + row, Dof<G> * block + col) = it.value();
        }
      }
    });
    ret.makeCompressed();
    return ret;
  }();

  /// @brief Sparse calculation of d2r_exp
  static void
  d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::d2r_exp_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

  /// @brief Sparse calculation of d2r_expinv
  static void
  d2r_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::d2r_expinv_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }
};

}  // namespace traits

}  // namespace smooth
