#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "impl/so3.hpp"
#include "internal/macro.hpp"
#include "lie_group_base.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SO2;
// \endcond

/**
 * @brief Base class for SO3 Lie group types
 *
 * Internally represented as a member of \f$ \mathbb{S}^3 \f$ (unit quaternions).
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [q_x, q_y, q_z, q_w] \f$ (same as Eigen quaternion).
 * - Tangent:  \f$ \mathbf{a} = [\omega_x, \omega_y, \omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$q_x^2 + q_y^2 + q_z^2 + q_w^2 = 1 \f$
 * - Tangent: \f$ -\pi < \omega_x, \omega_y, \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * 3x3 rotation matrix
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0 & -\omega_z & \omega_y \\
 *  \omega_z &   0 & -\omega_x \\
 *  -\omega_y &   \omega_x & 0 \\
 * \end{bmatrix}
 * \f]
 */
template<typename _Derived>
class SO3Base : public LieGroupBase<_Derived>
{
protected:
  using Base = LieGroupBase<_Derived>;  //!< Base class
  SO3Base()  = default;

public:

  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access quaterion.
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
   * @param i1, i2, i3 euler angle axis convention (\f$0=x, 1=y, 2=z\f$).
   *        Default values correspond to ZYX rotation.
   *
   * Returned angles a1, a2, a3 are s.t. rotation is described by
   * \f$ Rot_{i_1}(a_1) * Rot_{i_2}(a_2) * Rot_{i_3}(a_3). \f$
   * where \f$ Rot_i(a) \f$ rotates an angle \f$a\f$ around the \f$i\f$:th axis.
   */
  Eigen::Matrix<Scalar, 3, 1>
  eulerAngles(Eigen::Index i1 = 2, Eigen::Index i2 = 1, Eigen::Index i3 = 0) const
  {
    return quat().toRotationMatrix().eulerAngles(i1, i2, i3);
  }

  /**
   * @brief Rotation action on 3D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return quat() * v;
  }

  /**
   * @brief Project to SO2.
   *
   * This keeps the yaw/z axis component of the rotation.
   */
  SO2<Scalar> project_so2() const
  {
    return SO2<Scalar>(eulerAngles(0, 1, 2).z());
  }
};

// STORAGE TYPE TRAITS

// \cond
template<typename _Scalar>
class SO3;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<SO3<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SO3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO3<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SO3 Lie group.
 *
 * @see SO3Base for group API.
 */
template<typename _Scalar>
class SO3 : public SO3Base<SO3<_Scalar>>
{
  using Base = SO3Base<SO3<_Scalar>>;

  SMOOTH_GROUP_API(SO3);

public:
  /**
   * @brief Construct from quaternion.
   *
   * @param quat Eigen quaternion.
   *
   * @note Input is normalized inside constructor.
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & quat) : coeffs_(quat.normalized().coeffs())
  {}
};

using SO3f = SO3<float>;  //! SO3 with float scalar representation
using SO3d = SO3<double>;  //! SO3 with double scalar representation

}  // namespace smooth


// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO3<_Scalar>>>
  : public lie_traits<smooth::SO3<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SO3 Lie group.
 *
 * @see SO3Base for group API.
 */
template<typename _Scalar>
class Eigen::Map<smooth::SO3<_Scalar>>
  : public smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>
{
  using Base = smooth::SO3Base<Eigen::Map<smooth::SO3<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SO3<_Scalar>>>
  : public lie_traits<smooth::SO3<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SO3 Lie group.
 *
 * @see SO3Base for group API.
 */
template<typename _Scalar>
class Eigen::Map<const smooth::SO3<_Scalar>>
  : public smooth::SO3Base<Eigen::Map<const smooth::SO3<_Scalar>>>
{
  using Base = smooth::SO3Base<Eigen::Map<const smooth::SO3<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__SO3_HPP_
