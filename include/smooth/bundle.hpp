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

#ifndef SMOOTH__BUNDLE_HPP_
#define SMOOTH__BUNDLE_HPP_

#include "concepts.hpp"
#include "internal/bundle.hpp"
#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"

namespace smooth {

/**
 * @brief Base class for Bundle lie groups.
 *
 * Represents a direct product \f$ G_1 \times \ldots G_n \f$ between a
 * collection \f$ G_1, \ldots, G_n \f$ of Lie groups.
 *
 * Elements of the bundle are of the form \f$ (g_1, \ldots g_n) \f$, and
 * operations are performed element-wise, i.e. group composition is
 * \f[
 *   (g_1, \ldots, g_n) \circ (g_1', \ldots, g_n') = (g_1 \circ g_1', \ldots, g_n \circ g_n').
 * \f]
 *
 * The tangent space dimension is the sum of the tangent space dimension of the
 * member types, and similary for the reprenstation size and the matrix dimension.
 */
template<typename _Derived>
class BundleBase : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;
  using Impl = typename lie_traits<_Derived>::Impl;

protected:

  BundleBase() = default;

public:

  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Type of element in Bundle..
   */
  template<std::size_t Idx>
  using PartType = typename lie_traits<_Derived>::template PartPlainObject<Idx>;

  /**
   * @brief Access part no Idx of bundle.
   */
  template<std::size_t Idx>
  Eigen::Map<PartType<Idx>> part()
  requires is_mutable
  {
    return Eigen::Map<PartType<Idx>>(
      static_cast<_Derived &>(*this).data() + std::get<Idx>(Impl::RepSizesPsum));
  }

  /**
   * @brief Const access part no Idx of bundle.
   */
  template<std::size_t Idx>
  Eigen::Map<const PartType<Idx>> part() const
  {
    return Eigen::Map<const PartType<Idx>>(
      static_cast<const _Derived &>(*this).data() + std::get<Idx>(Impl::RepSizesPsum));
  }
};

/// Concept that is fulfilled by LieGroup or Eigen types
template<typename G>
concept LieGroupLike = requires
{
  typename lie_traits<G>::Impl;
  typename lie_traits<G>::Scalar;
};

// \cond
template<LieGroupLike... _Gs>
class Bundle;
// \endcond

// \cond
template<LieGroupLike... _Gs>
struct lie_traits<Bundle<_Gs...>>
{
  static constexpr bool is_mutable = true;

  using Impl   = BundleImpl<typename lie_traits<_Gs>::Impl...>;
  using Scalar = std::common_type_t<typename lie_traits<_Gs>::Scalar...>;

  static_assert((std::is_same_v<Scalar, typename lie_traits<_Gs>::Scalar> && ...),
    "Scalar type must be identical");

  template<typename NewScalar>
  using PlainObject = Bundle<typename lie_traits<_Gs>::template PlainObject<NewScalar>...>;

  template<std::size_t Idx>
  using PartPlainObject = typename lie_traits<
    std::tuple_element_t<Idx, std::tuple<_Gs...>>>::template PlainObject<Scalar>;
};
// \endcond

/**
 * @brief Storage implementation of Bundle lie group.
 *
 * @see BundleBase for details.
 */
template<LieGroupLike... _Gs>
class Bundle : public BundleBase<Bundle<_Gs...>>
{
  using Base = BundleBase<Bundle<_Gs...>>;

  SMOOTH_GROUP_API(Bundle);

public:
  /**
   * @brief Construct bundle from parts
   */
  template<typename... S>
  requires(std::is_assignable_v<_Gs, S> &&...)
  Bundle(S &&... gs)
  {
    auto tpl = std::forward_as_tuple(gs...);
    utils::static_for<sizeof...(_Gs)>(
      [this, &tpl](auto i) { Base::template part<i>() = std::get<i>(tpl); });
  }
};

}  // namespace smooth

// \cond
template<smooth::LieGroupLike... _Gs>
struct smooth::lie_traits<Eigen::Map<smooth::Bundle<_Gs...>>>
  : public lie_traits<smooth::Bundle<_Gs...>>
{};
// \endcond

/**
 * @brief Memory mapping of bundle Lie group.
 *
 * @see BundleBase for details.
 */
template<smooth::LieGroupLike... _Gs>
class Eigen::Map<smooth::Bundle<_Gs...>>
  : public smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>
{
  using Base = smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<smooth::LieGroupLike... _Gs>
struct smooth::lie_traits<Eigen::Map<const smooth::Bundle<_Gs...>>>
  : public lie_traits<smooth::Bundle<_Gs...>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of bundle Lie group.
 *
 * @see BundleBase for details.
 */
template<smooth::LieGroupLike... _Gs>
class Eigen::Map<const smooth::Bundle<_Gs...>>
  : public smooth::BundleBase<Eigen::Map<const smooth::Bundle<_Gs...>>>
{
  using Base = smooth::BundleBase<Eigen::Map<const smooth::Bundle<_Gs...>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__BUNDLE_HPP_
