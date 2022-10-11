// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Trait specialization to make std::variant<Manifold...> a Manifold.
 */

#include <variant>

#include "../concepts/manifold.hpp"

namespace smooth::traits {

/**
 * @brief Manifold model specification for std::variant<Manifold...>
 */
template<Manifold... Ms>
struct man<std::variant<Ms...>>
{
  // \cond
  using Scalar      = std::common_type_t<typename traits::man<Ms>::Scalar...>;
  using PlainObject = std::variant<Ms...>;
  template<typename NewScalar>
  using CastT = std::variant<typename traits::man<Ms>::template CastT<NewScalar>...>;

  static constexpr int Dof = -1;

  static inline Eigen::Index dof(const PlainObject & m)
  {
    const auto visitor = []<Manifold Mi>(const Mi & x) { return traits::man<Mi>::dof(x); };
    return std::visit(visitor, m);
  }

  static inline PlainObject Default(Eigen::Index dof)
  {
    return traits::man<std::tuple_element_t<0, std::tuple<Ms...>>>::Default(dof);
  }

  template<typename NewScalar>
  static inline CastT<NewScalar> cast(const PlainObject & m)
  {
    const auto visitor = []<Manifold Mi>(const Mi & x) -> CastT<NewScalar> {
      return traits::man<Mi>::template cast<NewScalar>(x);
    };
    return std::visit(visitor, m);
  }

  template<typename Derived>
  static inline PlainObject rplus(const PlainObject & m, const Eigen::MatrixBase<Derived> & a)
  {
    const auto visitor = [&a]<Manifold Mi>(const Mi & x) -> PlainObject { return traits::man<Mi>::rplus(x, a); };
    return std::visit(visitor, m);
  }

  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const PlainObject & m1, const PlainObject & m2)
  {
    const auto visitor = [&m2]<Manifold Mi>(const Mi & x) -> Eigen::VectorX<Scalar> {
      return traits::man<Mi>::rminus(x, std::get<Mi>(m2));
    };
    return std::visit(visitor, m1);
  }
  // \endcond
};

}  // namespace smooth::traits
