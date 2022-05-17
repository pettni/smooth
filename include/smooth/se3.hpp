// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <complex>

#include "detail/lie_group_base.hpp"
#include "detail/macro.hpp"
#include "detail/se3.hpp"
#include "lie_group.hpp"
#include "map.hpp"
#include "so3.hpp"

namespace smooth {

template<typename Scalar>
class SE2;

/**
 * @brief Base class for SE3 Lie group types.
 *
 * Internally represented as \f$\mathbb{S}^3 \times \mathbb{R}^3\f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [x, y, z, q_x, q_y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z] \f$
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
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *   R & T \\
 *   0 & 1
 * \end{bmatrix} \in \mathbb{R}^{4 \times 4}
 * \f]
 *
 * where \f$R\f$ is a 3x3 rotation matrix and \f$ T = [x, y, z]^T \f$.
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0        & -\omega_z & \omega_y  & v_x \\
 *  \omega_z  & 0         & -\omega_x & v_y \\
 *  -\omega_y & \omega_x  & 0         & v_y \\
 *  0         & 0         & 0         & 0
 * \end{bmatrix} \in \mathbb{R}^{4 \times 4}
 * \f]
 */
template<typename _Derived>
class SE3Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SE3Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access SO(3) part.
   */
  Map<SO3<Scalar>> so3() requires is_mutable
  {
    return Map<SO3<Scalar>>(static_cast<_Derived &>(*this).data() + 3);
  }

  /**
   * @brief Const access SO(3) part.
   */
  Map<const SO3<Scalar>> so3() const
  {
    return Map<const SO3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3);
  }

  /**
   * @brief Access R3 part.
   */
  Eigen::Map<Eigen::Vector3<Scalar>> r3() requires is_mutable
  {
    return Eigen::Map<Eigen::Vector3<Scalar>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Const access R3 part.
   */
  Eigen::Map<const Eigen::Vector3<Scalar>> r3() const
  {
    return Eigen::Map<const Eigen::Vector3<Scalar>>(static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Return as 3D Eigen transform.
   */
  Eigen::Transform<Scalar, 3, Eigen::Isometry> isometry() const
  {
    return Eigen::Translation<Scalar, 3>(r3()) * so3().quat();
  }

  /**
   * @brief Tranformation action on 3D vector.
   */
  template<typename EigenDerived>
  Eigen::Vector3<Scalar> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return so3() * v + r3();
  }

  /**
   * @brief Jacobian of rotation action w.r.t. group.
   *
   * \f[
   *   \mathrm{d}^r (X v)_X
   * \f]
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 6> dr_action(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    Eigen::Matrix<Scalar, 3, 6> ret;
    ret.template leftCols<3>()  = so3().matrix();
    ret.template rightCols<3>() = so3().dr_action(v);
    return ret;
  }

  /**
   * @brief Project to SE2.
   *
   * @note SE2 header must be included.
   */
  SE2<Scalar> project_se2() const
  {
    return SE2<Scalar>(so3().project_so2(), r3().template head<2>());
  }
};

// \cond
template<typename _Scalar>
class SE3;
// \endcond

// \cond
template<typename _Scalar>
struct liebase_info<SE3<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE3<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class SE3 : public SE3Base<SE3<_Scalar>>
{
  using Base = SE3Base<SE3<_Scalar>>;

  SMOOTH_GROUP_API(SE3);

public:
  /**
   * @brief Construct from SO3 and translation.
   *
   * @param so3 orientation component.
   * @param r3 translation component.
   */
  template<typename SO3Derived, typename T3Derived>
  SE3(const SO3Base<SO3Derived> & so3, const Eigen::MatrixBase<T3Derived> & r3)
  {
    Base::so3() = static_cast<const SO3Derived &>(so3);
    Base::r3()  = static_cast<const T3Derived &>(r3);
  }

  /**
   * @brief Construct from Eigen transform.
   */
  SE3(const Eigen::Transform<Scalar, 3, Eigen::Isometry> & t)
  {
    Base::so3() = smooth::SO3<Scalar>(Eigen::Quaternion<Scalar>(t.rotation()));
    Base::r3()  = t.translation();
  }
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<SE3<_Scalar>>> : public liebase_info<SE3<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class Map<SE3<_Scalar>> : public SE3Base<Map<SE3<_Scalar>>>
{
  using Base = SE3Base<Map<SE3<_Scalar>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<const SE3<_Scalar>>> : public liebase_info<SE3<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class Map<const SE3<_Scalar>> : public SE3Base<Map<const SE3<_Scalar>>>
{
  using Base = SE3Base<Map<const SE3<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using SE3f = SE3<float>;   ///< SE3 with float
using SE3d = SE3<double>;  ///< SE3 with double

}  // namespace smooth
