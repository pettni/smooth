#ifndef BUNDLE_HPP_
#define BUNDLE_HPP_

#include "impl/bundle.hpp"
#include "lie_group.hpp"
#include "macro.hpp"

namespace smooth {

// CRTP BASE

template<typename Derived>
class BundleBase : public LieGroupBase<Derived>
{
protected:
  using Base = LieGroupBase<Derived>;
  using Impl = typename lie_traits<Derived>::Impl;

  BundleBase() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  // BUNDLE API

  template<std::size_t Idx>
  using PartType = typename lie_traits<Derived>::template PartPlainObject<Idx>;

  /**
   * @brief Access part no Idx of bundle
   */
  template<std::size_t Idx>
  Eigen::Map<PartType<Idx>> part()
  {
    return Eigen::Map<PartType<Idx>>(
      static_cast<Derived &>(*this).data() + std::get<Idx>(Impl::RepSizesPsum));
  }

  /**
   * @brief Const access part no Idx of bundle
   */
  template<std::size_t Idx>
  Eigen::Map<const PartType<Idx>> part() const
  {
    return Eigen::Map<const PartType<Idx>>(
      static_cast<const Derived &>(*this).data() + std::get<Idx>(Impl::RepSizesPsum));
  }
};

template<typename... _Gs>
class Bundle;

// STORAGE TYPE TRAITS

template<typename... _Gs>
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

// STORAGE TYPE

template<typename... _Gs>
class Bundle : public BundleBase<Bundle<_Gs...>>
{
  using Base = BundleBase<Bundle<_Gs...>>;
  SMOOTH_GROUP_API(Bundle)
public:
  /**
   * @brief Construct bundle from parts
   */
  template<typename... S>
  requires(std::is_assignable_v<_Gs, S> &&...) Bundle(S &&... gs)
  {
    auto tpl = std::forward_as_tuple(gs...);
    utils::static_for<sizeof...(_Gs)>(
      [this, &tpl](auto i) { Base::template part<i>() = std::get<i>(tpl); });
  }
};

}  // namespace smooth

// MAP TYPE TRAITS

template<typename... _Gs>
struct smooth::lie_traits<Eigen::Map<smooth::Bundle<_Gs...>>>
    : public lie_traits<smooth::Bundle<_Gs...>>
{};

// MAP TYPE

template<typename... _Gs>
class Eigen::Map<smooth::Bundle<_Gs...>>
    : public smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>
{
  using Base = smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename... _Gs>
struct smooth::lie_traits<Eigen::Map<const smooth::Bundle<_Gs...>>>
    : public lie_traits<smooth::Bundle<_Gs...>>
{
  static constexpr bool is_mutable = false;
};

// CONST MAP TYPE

template<typename... _Gs>
class Eigen::Map<const smooth::Bundle<_Gs...>>
    : public smooth::BundleBase<Eigen::Map<const smooth::Bundle<_Gs...>>>
{
  using Base = smooth::BundleBase<Eigen::Map<const smooth::Bundle<_Gs...>>>;
  SMOOTH_CONST_MAP_API(Map)
};
#endif  // BUNDLE_HPP_
