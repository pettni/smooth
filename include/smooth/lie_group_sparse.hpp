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

#ifndef SMOOTH__LIE_GROUP_SPARSE_HPP_
#define SMOOTH__LIE_GROUP_SPARSE_HPP_

#include <Eigen/Sparse>

#include "internal/utils.hpp"
#include "lie_group.hpp"

namespace smooth {

/// @brief Forward-declare
// clang-format off
template<typename D> class C1Base;
template<typename D> class SE2Base;
template<typename D> class SE3Base;
template<typename D> class SO2Base;
template<typename D> class BundleBase;
// clang-format on

namespace traits {

/**
 * @brief Traits that defines sparsity patterns for various groups.
 */
template<typename G>
struct lie_sparse
{};

}  // namespace traits

////////////////////////////////////////////////////////
//// FREE FUNCTIONS FOR SPARSE LIE GROUP PROPERTIES ////
////////////////////////////////////////////////////////

/**
 * @brief Generators of lie algebra (inline variable).
 */
template<LieGroup G>
inline std::array<Eigen::SparseMatrix<Scalar<G>>, Dof<G>> generators_sparse =
  []() -> std::array<Eigen::SparseMatrix<Scalar<G>>, Dof<G>> {
  std::array<Eigen::SparseMatrix<Scalar<G>>, Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret[i] = ad<G>(Tangent<G>::Unit(i)).sparseView();
    ret[i].makeCompressed();
  }
  return ret;
}();

/**
 * @brief Sparsity pattern of ad (inline variable).
 */
template<LieGroup G>
inline Eigen::SparseMatrix<Scalar<G>> ad_sparse_pattern = [] {
  Eigen::SparseMatrix<Scalar<G>> ret(Dof<G>, Dof<G>);
  if constexpr (!IsCommutative<G>) {
    for (const auto & gen : generators_sparse<G>) { ret += gen; }
  }
  ret.makeCompressed();
  ret.coeffs().setZero();
  return ret;
}();

/**
 * @brief Sparse ad.
 *
 * @param[in, out] sp pre-allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 *
 * @see ad_sparse_pattern for a pre-allocated pattern.
 */
template<LieGroup G>
inline void ad_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a)
{
  assert(sp.isCompressed());
  sp.coeffs().setZero();
  for (auto k = 0u; k < Dof<G>; ++k) { sp += a(k) * generators_sparse<G>[k]; }
  assert(sp.isCompressed());
}

/**
 * @brief Sparsity pattern of dr_exp, dr_expinv (inline variable).
 */
template<LieGroup G>
inline Eigen::SparseMatrix<Scalar<G>> d_exp_sparse_pattern = [] {
  Eigen::SparseMatrix<Scalar<G>> ret(Dof<G>, Dof<G>);
  if constexpr (IsCommutative<G>) {
    // identity matrix
    for (auto i = 0u; i < Dof<G>; ++i) { ret.insert(i, i) = Scalar<G>(1); }
  } else if constexpr (requires { traits::lie_sparse<G>::d_exp_sparse_pattern; }) {
    // use specialized method if one exists
    ret = traits::lie_sparse<G>::d_exp_sparse_pattern;
  } else {
    // fall back on dense pattern
    for (auto i = 0u; i < Dof<G>; ++i) {
      for (auto j = 0u; j < Dof<G>; ++j) {
        ret.insert(i, j) = (i == j) ? Scalar<G>(1) : Scalar<G>(0);
      }
    }
  }
  ret.makeCompressed();
  return ret;
}();

/**
 * @brief Sparse dr_exp.
 *
 * @tparam G Lie group
 * @tparam Inv compute dr_expinv
 *
 * @param[in, out] sp pre-allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @see d_exp_sparse_pattern for a pre-allocated pattern.
 */
template<LieGroup G, bool Inv = false>
inline void
dr_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
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

/**
 * @brief Sparse dr_expinv.
 *
 * @tparam G Lie group
 *
 * @param[in, out] sp pre-allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @see d_exp_sparse_pattern for a pre-allocated pattern.
 */
template<LieGroup G>
inline void
dr_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
{
  dr_exp_sparse<G, true>(sp, a, i0);
}

/**
 * @brief Sparsity pattern of d2r_exp, d2r_expinv, and (inline variable).
 */
template<LieGroup G>
inline Eigen::SparseMatrix<Scalar<G>> d2_exp_sparse_pattern = [] {
  Eigen::SparseMatrix<Scalar<G>> ret(Dof<G>, Dof<G> * Dof<G>);
  if constexpr (IsCommutative<G>) {
    // zero matrix--do nothing
  } else if constexpr (requires { traits::lie_sparse<G>::d2_exp_sparse_pattern; }) {
    // use spacialized method
    ret = traits::lie_sparse<G>::d2_exp_sparse_pattern;
  } else {
    // dense pattern
    ret = Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>>::Ones().sparseView();
  }
  ret.makeCompressed();
  return ret;
}();

/**
 * @brief Sparse d2r_exp.
 *
 * @tparam G Lie group
 * @tparam Inv compute d2r_expinv
 *
 * @param[in, out] sp pre-allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @see d2_exp_sparse_pattern for a pre-allocated pattern.
 */
template<LieGroup G, bool Inv = false>
inline void
d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
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

/**
 * @brief Sparse d2r_expinv.
 *
 * @tparam G Lie group
 *
 * @param[in, out] sp pre-allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @see d2_exp_sparse_pattern for a pre-allocated pattern.
 */
template<LieGroup G>
inline void
d2r_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
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
    ret.insert(2, 0)     = 0;
    ret.insert(2, 1)     = 0;
    ret.insert(2, 2)     = 0;
    ret.insert(0, 2)     = 0;
    ret.insert(1, 2)     = 0;
    ret.middleCols(3, 3) = ret.leftCols(3);
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

  static void
  dr_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::dr_exp_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

  static void
  dr_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::dr_expinv_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

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

  static void
  d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0)
  {
    utils::static_for<G::BundleSize>([&](auto I) {
      static constexpr auto Dof0 = G::template PartStart<I>;
      ::smooth::d2r_exp_sparse<typename G::template PartType<I>>(
        sp, a.template segment<G::template PartDof<I>>(Dof0), i0 + Dof0);
    });
  }

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

#endif  // SMOOTH__LIE_GROUP_SPARSE_HPP_
