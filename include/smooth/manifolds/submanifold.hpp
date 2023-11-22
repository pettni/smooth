// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <algorithm>

#include "smooth/concepts/manifold.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief A Submanifold is a subspace of a manifold defined by an origin m0 and a tangent subspace.
 */
template<Manifold M>
class SubManifold
{
public:
  /**
   * @brief Construct with full tangent space.
   */
  SubManifold() : SubManifold(traits::man<M>::Default(), traits::man<M>::Default()) {}

  /**
   * @brief Create a sub-manifold of M defined as
   * \[
   *      x \in M : x = x0 \oplus \sum_k \alpha_k e_{i_k},  \alpha_k \in R
   * \]
   * where k are the "active" dimensions.
   *
   * @param m0 reference point
   * @param m current value
   * @param fixed_dims active dimensions
   *
   * @note Binary operations on SubManifold's require that both instance
   *       have identical reference points and active dimensions.
   */
  SubManifold(const M & m0, const M & m, Eigen::Ref<const Eigen::VectorXi> fixed_dims = Eigen::VectorXi::Zero(0))
      : m_m0(m0), m_m(m), m_fixed_dims(fixed_dims)
  {
    [[maybe_unused]] const auto in_range_pred = [&](auto x) { return x >= 0 && x < ::smooth::dof(m_m0); };
    assert(std::ranges::all_of(fixed_dims, in_range_pred));
    std::sort(m_fixed_dims.begin(), m_fixed_dims.end());
    m_calc.setZero(::smooth::dof(m_m0));
  }

  /**
   * @brief As above, but with m = m0.
   */
  SubManifold(const M & m0, Eigen::Ref<const Eigen::VectorXi> fixed_dims) : SubManifold(m0, m0, fixed_dims) {}

  /// @brief Access value in embedded space
  const M & m() const { return m_m; }

  /// @brief Access origin in embedded space
  const M & m0() const { return m_m0; }

  /// @brief Access active dimensions
  const Eigen::VectorXi & fixed_dims() const { return m_fixed_dims; }

  /// @brief Degrees of freedom
  Eigen::Index dof() const { return ::smooth::dof(m_m0) - m_fixed_dims.size(); }

  /// @brief Right-plus
  template<typename Derived>
  SubManifold<M> rplus(const Eigen::MatrixBase<Derived> & a) const
  {
    assert(dof() == a.size());
    m_calc.setZero(::smooth::dof(m_m0));
    for (auto i = 0, j = 0, k = 0; i < m_calc.size(); ++i) {
      // i: full range
      // j: reduced range
      // k: idx in fixed_dims
      if (k >= m_fixed_dims.size() || i != m_fixed_dims(k)) {
        m_calc(i) = a(j++);
      } else {
        ++k;
      }
    }
    return SubManifold<M>(m_m0, traits::man<M>::rplus(m_m, m_calc), m_fixed_dims);
  }

  /// @brief Right-minus
  Eigen::Vector<typename traits::man<M>::Scalar, -1> rminus(const SubManifold<M> & other) const
  {
    assert(m_fixed_dims.isApprox(other.fixed_dims()));
    assert(m_m0.isApprox(other.m0()));

    m_calc = traits::man<M>::rminus(m_m, other.m());

    Eigen::Vector<typename traits::man<M>::Scalar, -1> ret;
    ret.setZero(dof());
    for (auto i = 0, j = 0, k = 0; i < m_calc.size(); ++i) {
      // i: full range
      // j: reduced range
      // k: idx in fixed_dims
      if (k >= m_fixed_dims.size() || i != m_fixed_dims(k)) {
        ret(j++) = m_calc(i);
      } else {
        ++k;
      }
    }
    return ret;
  }

private:
  M m_m0{};
  M m_m{};
  Eigen::VectorXi m_fixed_dims{};

  // calculation vector
  mutable Tangent<M> m_calc{};
};

namespace traits {

/// @brief Manifold specialization for SubManifold
template<Manifold M>
struct man<SubManifold<M>>
{
  using Scalar      = man<M>::Scalar;
  using PlainObject = SubManifold<M>;
  template<typename NewScalar>
  using CastT = SubManifold<typename man<M>::template CastT<NewScalar>>;

  static constexpr int Dof = -1;

  static inline Eigen::Index dof(const PlainObject & m) { return m.dof(); }

  static inline PlainObject Default(Eigen::Index dof)
  {
    Eigen::VectorXi fixed_dims = Eigen::VectorXi::Zero(0);
    return PlainObject(man<M>::Default(dof));
  }

  template<typename NewScalar>
  static inline CastT<NewScalar> cast(const PlainObject & m)
  {
    return CastT<NewScalar>(
      man<M>::template cast<NewScalar>(m.m()), man<M>::template cast<NewScalar>(m.m0()), m.fixed_dims());
  }

  template<typename Derived>
  static inline PlainObject rplus(const PlainObject & m, const Eigen::MatrixBase<Derived> & a)
  {
    return m.rplus(a);
  }

  static inline Eigen::Vector<Scalar, Dof> rminus(const PlainObject & m1, const PlainObject & m2)
  {
    return m1.rminus(m2);
  }
};

}  // namespace traits

SMOOTH_END_NAMESPACE
