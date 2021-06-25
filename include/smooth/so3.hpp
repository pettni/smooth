#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "impl/so3.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"

namespace smooth {

template<typename Scalar>
class SO2;

// CRTP BASE

/**
 * @brief SO3 Lie Group represented as S3
 *
 * Memory layout
 * =============
 * Group:    qx qy qz qw  (same as Eigen quaternion)
 * Tangent:  wx wy wz
 *
 * Lie group Matrix form
 * =====================
 *
 * 3x3 rotation matrix
 *
 * Lie algebra Matrix form
 * =====================
 *
 * [  0 -Ωz  Ωy ]
 * [  Ωz  0 -Ωx ]
 * [ -Ωy Ωx   0 ]
 *
 * Constraints
 * ===========
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx, Ωy, Ωz <= pi
 */
template<typename _Derived>
class SO3Base : public LieGroupBase<_Derived>
{
protected:
  using Base = LieGroupBase<_Derived>;
  SO3Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  /**
   * @brief Access quaterion
   */
  Eigen::Map<Eigen::Quaternion<Scalar>> quat()
  requires is_mutable
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Access const quaterion.
   */
  Eigen::Map<const Eigen::Quaternion<Scalar>> quat() const
  {
    return Eigen::Map<const Eigen::Quaternion<Scalar>>(static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Return euler angles.
   *
   * @param i1, i2, i3 euler angle axis convention (0=x, 1=y, 2=z).
   *        Default values correspond to ZYX rotation.
   *
   * Returned angles a1, a2, a3 are s.t. rotation is described by
   * Rot_i1(a1) * Rot_i2(a2) * Rot_i3(a3),
   */
  Eigen::Matrix<Scalar, 3, 1> eulerAngles(
    Eigen::Index i1 = 2, Eigen::Index i2 = 1, Eigen::Index i3 = 0) const
  {
    return quat().toRotationMatrix().eulerAngles(i1, i2, i3);
  }

  /**
   * @brief Rotation action on 3D vector.
   *
   * @param v 3D vector to rotate
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return quat() * v;
  }

  /**
   * @brief Project to SO2.
   *
   * This keeps the "z"/yaw component of the rotation.
   */
  SO2<Scalar> project_so2() const
  {
    return SO2<Scalar>(eulerAngles(0, 1, 2).z());
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SO3;

template<typename _Scalar>
struct lie_traits<SO3<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SO3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO3<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SO3 : public SO3Base<SO3<_Scalar>>
{
  using Base = SO3Base<SO3<_Scalar>>;
  SMOOTH_GROUP_API(SO3)
public:
  /**
   * @brief Construct from quaternion.
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & quat) : coeffs_(quat.normalized().coeffs())
  {}
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

}  // namespace smooth

// MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO3<_Scalar>>>
  : public lie_traits<smooth::SO3<_Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SO3<_Scalar>>
  : public smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>
{
  using Base = smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SO3<_Scalar>>>
  : public lie_traits<smooth::SO3<_Scalar>>
{
  static constexpr bool is_mutable = false;
};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SO3<_Scalar>>
  : public smooth::SO3Base<Eigen::Map<const smooth::SO3<_Scalar>>>
{
  using Base = smooth::SO3Base<Eigen::Map<const smooth::SO3<_Scalar>>>;
  SMOOTH_CONST_MAP_API(Map)
};

#endif  // SMOOTH__SO3_HPP_
