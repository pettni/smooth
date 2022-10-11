// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "../concepts/lie_group.hpp"
#include "../detail/traits.hpp"

/**
 * @file
 * @brief Trait specialization to make Eigen vectors LieGroups.
 */

namespace smooth::traits {

/**
 * @brief LieGroup model specification for RnType
 */
template<RnType G>
struct lie<G>
{
  // \cond
  static constexpr int Dof            = G::SizeAtCompileTime;
  static constexpr bool IsCommutative = true;

  using Scalar      = typename G::Scalar;
  using PlainObject = Eigen::Vector<Scalar, Dof>;
  template<typename NewScalar>
  using CastT = Eigen::Vector<NewScalar, Dof>;

  // group interface

  static inline PlainObject Identity(Eigen::Index dof) { return G::Zero(dof); }
  static inline PlainObject Random(Eigen::Index dof) { return G::Random(dof); }
  static inline Eigen::Matrix<Scalar, Dof, Dof> Ad(const G & g)
  {
    return Eigen::Matrix<Scalar, Dof, Dof>::Identity(g.size(), g.size());
  }
  template<typename Derived>
  static inline PlainObject composition(const G & g1, const Eigen::MatrixBase<Derived> & g2)
  {
    return g1 + g2;
  }
  static inline Eigen::Index dof(const G & g) { return g.size(); }
  static inline PlainObject inverse(const G & g) { return -g; }
  template<typename Derived>
  static inline bool isApprox(const G & g, const Eigen::MatrixBase<Derived> & g2, Scalar eps)
  {
    return g.isApprox(g2, eps);
  }
  static inline Eigen::Vector<Scalar, Dof> log(const G & g) { return g; }
  template<typename NewScalar>
  static inline Eigen::Vector<NewScalar, Dof> cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static inline Eigen::Matrix<Scalar, Dof, Dof> ad(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix<Scalar, Dof, Dof>::Zero(a.size(), a.size());
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a;
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, Dof, Dof> dr_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix<Scalar, Dof, Dof>::Identity(a.size(), a.size());
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, Dof, Dof> dr_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix<Scalar, Dof, Dof>::Identity(a.size(), a.size());
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, Dof, (Dof > 0 ? Dof * Dof : -1)> d2r_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix<Scalar, Dof, (Dof > 0 ? Dof * Dof : -1)>::Zero(a.size(), a.size() * a.size());
  }
  template<typename Derived>
  static inline Eigen::Matrix<Scalar, Dof, (Dof > 0 ? Dof * Dof : -1)> d2r_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix<Scalar, Dof, (Dof > 0 ? Dof * Dof : -1)>::Zero(a.size(), a.size() * a.size());
  }
  // \endcond
};

}  // namespace smooth::traits
