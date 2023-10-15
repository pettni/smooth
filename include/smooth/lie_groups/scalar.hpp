// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Trait specialization to make scalars LieGroups.
 */

#include "../concepts/lie_group.hpp"
#include "../detail/traits.hpp"

SMOOTH_BEGIN_NAMESPACE

namespace traits {
/**
 * @brief LieGroup model specification for ScalarType
 */
template<ScalarType G>
struct lie<G>
{
  // \cond
  using Scalar      = G;
  using PlainObject = G;
  template<typename NewScalar>
  using CastT = NewScalar;

  static constexpr int Dof            = 1;
  static constexpr bool IsCommutative = true;

  // group interface

  static inline PlainObject Identity(Eigen::Index) { return G(0); }
  static inline PlainObject Random(Eigen::Index)
  {
    return G(Scalar(-1) + static_cast<Scalar>(rand()) / static_cast<Scalar>(RAND_MAX / 2));
  }
  static inline Eigen::Matrix<Scalar, 1, 1> Ad(G) { return Eigen::Matrix<Scalar, 1, 1>{1}; }
  static inline PlainObject composition(G g1, G g2) { return g1 + g2; }
  static inline Eigen::Index dof(G) { return 1; }
  static inline PlainObject inverse(G g) { return -g; }
  static inline bool isApprox(G g1, G g2, Scalar eps)
  {
    using std::abs;
    return abs<G>(g1 - g2) <= eps * abs<G>(g1);
  }
  static inline Eigen::Matrix<Scalar, 1, 1> log(G g) { return Eigen::Matrix<Scalar, 1, 1>{g}; }
  template<typename NewScalar>
  static inline NewScalar cast(G g)
  {
    return static_cast<NewScalar>(g);
  }

  // tangent interface

  template<typename Derived>
  static inline Eigen::Matrix<Scalar, 1, 1> ad(const Eigen::MatrixBase<Derived> &)
  {
    return Eigen::Matrix<Scalar, 1, 1>::Zero();
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a(0);
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, 1, 1> dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return Eigen::Matrix<Scalar, 1, 1>::Identity();
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, 1, 1> dr_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return Eigen::Matrix<Scalar, 1, 1>::Identity();
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, 1, 1> d2r_exp(const Eigen::MatrixBase<Derived> &)
  {
    return Eigen::Matrix<Scalar, 1, 1>::Zero();
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, 1, 1> d2r_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return Eigen::Matrix<Scalar, 1, 1>::Zero();
  }
  // \endcond
};

}  // namespace traits

SMOOTH_END_NAMESPACE
