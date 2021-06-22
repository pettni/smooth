#ifndef SMOOTH__SE3_HPP_
#define SMOOTH__SE3_HPP_

#include <Eigen/Core>

#include <complex>

#include "impl/se3.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"
#include "so3.hpp"

namespace smooth {

// CRTP BASE

template<typename _Derived>
class SE3Base : public LieGroupBase<_Derived>
{
protected:
  using Base = LieGroupBase<_Derived>;
  SE3Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  /**
   * Access SO(3) part
   */
  Eigen::Map<SO3<Scalar>> so3()
  requires is_mutable
  {
    return Eigen::Map<SO3<Scalar>>(static_cast<_Derived &>(*this).data() + 3);
  }

  /**
   * Const access SO(3) part
   */
  Eigen::Map<const SO3<Scalar>> so3() const
  {
    return Eigen::Map<const SO3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3);
  }

  /**
   * Access T(3) part
   */
  Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> t3()
  requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 3, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * Const access T(3) part
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t3() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * Tranformation action on 3D vector
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v)
  {
    return so3().matrix() * v + t3();
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SE3;

template<typename _Scalar>
struct lie_traits<SE3<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE3<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SE3 : public SE3Base<SE3<_Scalar>>
{
  using Base = SE3Base<SE3<_Scalar>>;
  SMOOTH_GROUP_API(SE3)
public:
  /**
   * @brief Construct from SO3 and translation
   */
  template<typename SO3Derived, typename T3Derived>
  SE3(const SO3Base<SO3Derived> & so3, const Eigen::MatrixBase<T3Derived> & t3)
  {
    Base::so3() = so3;
    Base::t3()  = t3;
  }
};

using SE3f = SE3<float>;
using SE3d = SE3<double>;

}  // namespace smooth

// MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SE3<_Scalar>>>
  : public lie_traits<smooth::SE3<_Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SE3<_Scalar>>
  : public smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>
{
  using Base = smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SE3<_Scalar>>>
  : public lie_traits<smooth::SE3<_Scalar>>
{
  static constexpr bool is_mutable = false;
};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SE3<_Scalar>>
  : public smooth::SE3Base<Eigen::Map<const smooth::SE3<_Scalar>>>
{
  using Base = smooth::SE3Base<Eigen::Map<const smooth::SE3<_Scalar>>>;
  SMOOTH_CONST_MAP_API(Map)
};

#endif  // SMOOTH__SE3_HPP_
