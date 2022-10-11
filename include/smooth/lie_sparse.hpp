// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Sparse>

#include "detail/utils.hpp"
#include "lie_groups.hpp"

/**
 * @file
 * @brief Sparse versions of certain Lie group methods.
 *
 * Implemented methods contain two parts: a pattern and the function itself. The pattern is in the
 * form of an inline sparse matrix variable that is pre-allocated with the appropriate nonzeros.
 *
 * The intended usage pattern is to copy a pattern and use the resulting variable in calls to these
 * methods. That way no additional  allocation is needed.
 *
 * Example:
 * @code
 * auto my_var = ad_sparse_pattern;  // copy pattern
 * ad_sparse(my_var, a);             // update my_var in-place without allocation
 * @endcode
 */

namespace smooth {

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
 * @brief Sparsity pattern of ad_sparse (inline variable).
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
 * @param[in, out] sp allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 *
 * @warning sp must be pre-allocated with the appropriate nonzeros and compressed
 *
 * @see ad_sparse_pattern() for a pre-allocated pattern.
 */
template<LieGroup G>
void ad_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a);

/**
 * @brief Sparsity pattern of dr_exp_sparse, dr_expinv_sparse (inline variable).
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
      for (auto j = 0u; j < Dof<G>; ++j) { ret.insert(i, j) = (i == j) ? Scalar<G>(1) : Scalar<G>(0); }
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
 * @param[in, out] sp allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @warning sp must be pre-allocated with the appropriate nonzeros and compressed
 *
 * @see d_exp_sparse_pattern() for a pre-allocated pattern.
 */
template<LieGroup G, bool Inv = false>
void dr_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0);

/**
 * @brief Sparse dr_expinv.
 *
 * @tparam G Lie group
 *
 * @param[in, out] sp allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @warning sp must be pre-allocated with the appropriate nonzeros and compressed
 *
 * @see d_exp_sparse_pattern() for a pre-allocated pattern.
 */
template<LieGroup G>
void dr_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0);

/**
 * @brief Sparsity pattern of d2r_exp_sparse(), d2r_expinv_sparse() (inline variable).
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
 * @param[in, out] sp allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @warning sp must be pre-allocated with the appropriate nonzeros and compressed
 *
 * @see d2_exp_sparse_pattern() for a pre-allocated pattern.
 */
template<LieGroup G, bool Inv = false>
void d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0);

/**
 * @brief Sparse d2r_expinv.
 *
 * @tparam G Lie group
 *
 * @param[in, out] sp allocated and compressed sparse matrix
 * @param[in] a Lie algebra element
 * @param[in] i0 block index (row and column) in sp where result is inserted
 *
 * @warning sp must be pre-allocated with the appropriate nonzeros and compressed
 *
 * @see d2_exp_sparse_pattern() for a pre-allocated pattern.
 */
template<LieGroup G>
inline void d2r_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & sp, const Tangent<G> & a, Eigen::Index i0 = 0);

}  // namespace smooth

#include "detail/lie_group_sparse_impl.hpp"
