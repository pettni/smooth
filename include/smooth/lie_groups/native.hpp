// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Trait specialization to make native Lie groups LieGroups.
 */

#include "smooth/concepts/lie_group.hpp"

SMOOTH_BEGIN_NAMESPACE

namespace traits {
/**
 * @brief Concept defining class with an internal Lie group interface.
 *
 * Concept satisfied if G has members that correspond to the LieGroup concept.
 */
template<typename G>
concept NativeLieGroup =
  requires {
    // clang-format off
  typename G::Scalar;
  typename G::Tangent;
  typename G::PlainObject;
  {G::Dof}->std::convertible_to<int>;
  {G::IsCommutative}->std::convertible_to<bool>;
} &&
(!(G::Dof > 0) || requires {
  {G::Identity()}->std::convertible_to<typename G::PlainObject>;
  {G::Random()}->std::convertible_to<typename G::PlainObject>;
}) &&
(!(G::Dof == -1) || requires (Eigen::Index dof) {  //NOLINT
  {G::Identity(dof)}->std::convertible_to<typename G::PlainObject>;
  {G::Random(dof)}->std::convertible_to<typename G::PlainObject>;
}) &&
(G::Tangent::SizeAtCompileTime == G::Dof) &&
requires(const G & g1, const G & g2, typename G::Scalar eps) {
  {g1.dof()}->std::convertible_to<Eigen::Index>;  // degrees of freedom at runtime
  {g1.Ad()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {g1 * g2}->std::convertible_to<typename G::PlainObject>;
  {g1.inverse()}->std::convertible_to<typename G::PlainObject>;
  {g1.isApprox(g2, eps)}->std::convertible_to<bool>;
  {g1.log()}->std::convertible_to<Eigen::Vector<typename G::Scalar, G::Dof>>;
} &&
requires(const Eigen::Vector<typename G::Scalar, G::Dof> & a) {
  {G::ad(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::exp(a)}->std::convertible_to<typename G::PlainObject>;
  {G::dr_exp(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::dr_expinv(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::d2r_exp(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof * G::Dof>>;
  {G::d2r_expinv(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof * G::Dof>>;
    // clang-format on
  };  // NOLINT

/**
 * @brief LieGroup model specification for NativeLieGroup
 */
template<NativeLieGroup G>
struct lie<G>
{
  // \cond
  using Scalar = typename G::Scalar;
  template<typename NewScalar>
  using CastT       = typename G::template CastT<NewScalar>;
  using PlainObject = typename G::PlainObject;

  static constexpr int Dof            = G::Dof;
  static constexpr bool IsCommutative = G::IsCommutative;

  // group interface

  static inline PlainObject Identity([[maybe_unused]] Eigen::Index dof)
  {
    if constexpr (G::Dof == -1) {
      return G::Identity(dof);
    } else {
      return G::Identity();
    }
  }
  static inline PlainObject Random([[maybe_unused]] Eigen::Index dof)
  {
    if constexpr (G::Dof == -1) {
      return G::Random(dof);
    } else {
      return G::Random();
    }
  }
  static inline typename G::TangentMap Ad(const G & g) { return g.Ad(); }
  template<NativeLieGroup Go>
  static inline PlainObject composition(const G & g1, const Go & g2)
  {
    return g1.operator*(g2);
  }
  static inline Eigen::Index dof(const G &) { return G::Dof; }
  static inline PlainObject inverse(const G & g) { return g.inverse(); }
  template<NativeLieGroup Go>
  static inline bool isApprox(const G & g, const Go & go, Scalar eps)
  {
    return g.isApprox(go, eps);
  }
  static inline typename G::Tangent log(const G & g) { return g.log(); }
  template<typename NewScalar>
  static inline CastT<NewScalar> cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static inline typename G::TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  {
    return G::ad(a);
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::exp(a);
  }
  template<typename Derived>
  static inline typename G::TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_exp(a);
  }
  template<typename Derived>
  static inline typename G::TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_expinv(a);
  }
  template<typename Derived>
  static inline typename G::Hessian d2r_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::d2r_exp(a);
  }
  template<typename Derived>
  static inline typename G::Hessian d2r_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return G::d2r_expinv(a);
  }
  // \endcond
};

}  // namespace traits

SMOOTH_END_NAMESPACE
