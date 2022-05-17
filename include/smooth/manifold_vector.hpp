// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Trait specialization to make std::vector<Manifold> a Manifold.
 *
 * TODO(pettni) remove class and just use std::vector
 */

#include <Eigen/Sparse>

#include <numeric>
#include <ranges>

#include "detail/utils.hpp"
#include "manifold.hpp"

namespace smooth {

/**
 * @brief Manifold interface for std::vector<Manifold>
 */
template<Manifold M>
struct traits::man<std::vector<M>>
{
  // \cond
  using Scalar      = ::smooth::Scalar<M>;
  using PlainObject = std::vector<M>;
  template<typename NewScalar>
  using CastT = std::vector<typename man<M>::template CastT<NewScalar>>;

  static constexpr int Dof = -1;

  static inline Eigen::Index dof(const PlainObject & m)
  {
    if constexpr (traits::man<M>::Dof > 0) {
      return static_cast<Eigen::Index>(m.size()) * traits::man<M>::Dof;
    } else {
      return std::accumulate(m.begin(), m.end(), 0u, [](auto s, const auto & item) {
        return s + traits::man<M>::dof(item);
      });
    }
  }

  static inline PlainObject Default(Eigen::Index dof)
  {
    /// @note If underlying M has dynamic size, mdof not uniquely defined
    const Eigen::Index mdof = ::smooth::Dof<M> != -1 ? ::smooth::Dof<M> : 1;
    const auto size         = static_cast<std::size_t>(dof / mdof);

    return PlainObject(size, traits::man<M>::Default(mdof));
  }

  template<typename NewScalar>
  static inline auto cast(const PlainObject & m)
  {
    const auto transformer = [](const M & mi) ->
      typename traits::man<M>::template CastT<NewScalar> {
        return traits::man<M>::template cast<NewScalar>(mi);
      };
    const auto casted_view = m | std::views::transform(transformer);
    return std::vector(std::ranges::begin(casted_view), std::ranges::end(casted_view));
  }

  template<typename Derived>
  static inline PlainObject rplus(const PlainObject & m, const Eigen::MatrixBase<Derived> & a)
  {
    PlainObject m_plus_a;
    m_plus_a.reserve(m.size());
    for (Eigen::Index dof_cntr = 0; const auto & mi : m) {
      const auto dof_i = traits::man<M>::dof(mi);
      m_plus_a.push_back(
        traits::man<M>::rplus(mi, a.template segment<traits::man<M>::Dof>(dof_cntr, dof_i)));
      dof_cntr += dof_i;
    }
    return m_plus_a;
  }

  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const PlainObject & m1, const PlainObject & m2)
  {
    Eigen::Index dof_cnts = 0;
    if (traits::man<M>::Dof > 0) {
      dof_cnts = static_cast<Eigen::Index>(traits::man<M>::Dof * m1.size());
    } else {
      for (auto i = 0u; i != m1.size(); ++i) { dof_cnts += ::smooth::dof<M>(m1[i]); }
    }

    Eigen::VectorX<Scalar> ret(dof_cnts);

    for (Eigen::Index idx = 0; const auto & [m1i, m2i] : utils::zip(m1, m2)) {
      const auto & size_i = ::smooth::dof<M>(m1i);

      ret.template segment<traits::man<M>::Dof>(idx, size_i) = traits::man<M>::rminus(m1i, m2i);
      idx += size_i;
    }

    return ret;
  }
  // \endcond
};

}  // namespace smooth
