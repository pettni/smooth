#ifndef BUNDLE_HPP_
#define BUNDLE_HPP_

#include "impl/bundle.hpp"
#include "lie_group.hpp"

namespace smooth {

// CRTP BASE

template<typename Derived>
class BundleBase : public LieGroup<Derived>
{
protected:
  using Base = LieGroup<Derived>;
  using Impl = typename lie_traits<Derived>::Impl;

  BundleBase() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  // BUNDLE API

  template<std::size_t Idx>
  using PartType = typename lie_traits<Derived>::template PartPlainObject<Idx>;

  template<std::size_t Idx>
  Eigen::Map<PartType<Idx>> part()
  {
    return Eigen::Map<PartType<Idx>>(Base::data() + std::get<Idx>(Impl::RepSizesPsum));
  }

  template<std::size_t Idx>
  Eigen::Map<const PartType<Idx>> part() const
  {
    return Eigen::Map<const PartType<Idx>>(Base::data() + std::get<Idx>(Impl::RepSizesPsum));
  }
};

template<typename... Gs>
class Bundle;

// STORAGE TYPE TRAITS

template<typename... _Gs>
struct lie_traits<Bundle<_Gs...>>
{
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

template<typename... Gs>
class Bundle : public BundleBase<Bundle<Gs...>>
{
  using Base = BundleBase<Bundle<Gs...>>;

public:
  SMOOTH_GROUP_CONSTUCTORS(Bundle)
  SMOOTH_INHERIT_TYPEDEFS

  // REQUIRED API

  using Storage = Eigen::Matrix<Scalar, RepSize, 1>;

  Storage & coeffs() { return coeffs_; }

  const Storage & coeffs() const { return coeffs_; }

private:
  Storage coeffs_;
};

}  // namespace smooth

// MAP TYPE TRAITS

template<typename... _Gs>
struct smooth::lie_traits<Eigen::Map<smooth::Bundle<_Gs...>>> : public lie_traits<Bundle<_Gs...>>
{};

// MAP TYPE

template<typename... _Gs>
class Eigen::Map<smooth::Bundle<_Gs...>>
    : public smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>
{
  using Base = smooth::BundleBase<Eigen::Map<smooth::Bundle<_Gs...>>>;

public:
  SMOOTH_INHERIT_TYPEDEFS

  Map(Scalar * p) : coeffs_(p) {}

  // REQUIRED API

  using Storage = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>>;

  Storage & coeffs() { return coeffs_; }

  const Storage & coeffs() const { return coeffs_; }

private:
  Storage coeffs_;
};

#endif  // BUNDLE_HPP_
