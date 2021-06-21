#ifndef SO3_HPP_
#define SO3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "impl/so3.hpp"
#include "lie_group.hpp"

namespace smooth {

// CRTP BASE

template<typename Derived>
class SO3Base : public LieGroup<Derived>
{
protected:
  using Base = LieGroup<Derived>;
  SO3Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  // SO3 API

  Eigen::Map<Eigen::Quaternion<Scalar>> & quat()
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(Base::coeffs().data());
  }

  Eigen::Map<const Eigen::Quaternion<Scalar>> & quat() const
  {
    return Eigen::Map<const Eigen::Quaternion<Scalar>>(Base::coeffs().data());
  }
};

// STORAGE TYPE TRAITS

template<typename Scalar>
class SO3;

template<typename _Scalar>
struct lie_traits<SO3<_Scalar>>
{
  using Impl   = SO3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO3<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SO3 : public SO3Base<SO3<_Scalar>>
{
  using Base = typename SO3Base<SO3<_Scalar>>::Base;

public:
  SMOOTH_GROUP_CONSTUCTORS(SO3)
  SMOOTH_INHERIT_TYPEDEFS

  // REQUIRED API

  using Storage = Eigen::Matrix<Scalar, RepSize, 1>;

  Storage & coeffs() { return coeffs_; }

  const Storage & coeffs() const { return coeffs_; }

  // SO3 API

  // Construct from quaternion
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & quat) : coeffs_(quat.coeffs())
  {}

private:
  friend Base;
  Storage coeffs_;
};

}  // namespace smooth

// MAP TYPE TRAITS

template<typename Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO3<Scalar>>> : public lie_traits<smooth::SO3<Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SO3<_Scalar>> : public smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>
{
  using Base = typename smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>::Base;

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

#endif  // SO3_HPP_
