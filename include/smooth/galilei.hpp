// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "detail/galilei.hpp"
#include "detail/lie_group_base.hpp"
#include "detail/macro.hpp"
#include "lie_group.hpp"
#include "map.hpp"
#include "so3.hpp"

namespace smooth {

/**
 * @brief Base class for Galielei Lie group types.
 *
 * Internally represented as \f$\mathbb{S}^3 \times \mathbb{R}^3 \times \mathbb{R}^3 \times
 * \mathbb{R} \f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [vx, vy, vz, px, py, pz, t q_x, q_y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [bx, by, bz, tx, ty, tz, s, \omega_x, \omega_y, \omega_z] \f$
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
 *   R & v & p \\
 *   0 & 1 & t \\
 *   0 & 0 & 1
 * \end{bmatrix} \in \mathbb{R}^{5 \times 5}
 * \f]
 *
 * where \f$R\f$ is a 3x3 rotation matrix.
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0        & -\omega_z & \omega_y  & b_x & t_x\\
 *  \omega_z  & 0         & -\omega_x & b_y & t_y\\
 *  -\omega_y & \omega_x  & 0         & b_y & t_y\\
 *  0         & 0         & 0         & 0   &   s\\
 *  0         & 0         & 0         & 0   &   1\\
 * \end{bmatrix} \in \mathbb{R}^{5 \times 5}
 * \f]
 */
template<typename _Derived>
class GalileiBase : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  GalileiBase() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access SO(3) part.
   */
  Map<SO3<Scalar>> so3() requires is_mutable { return Map<SO3<Scalar>>(static_cast<_Derived &>(*this).data() + 7); }

  /**
   * @brief Const access SO(3) part.
   */
  Map<const SO3<Scalar>> so3() const { return Map<const SO3<Scalar>>(static_cast<const _Derived &>(*this).data() + 7); }

  /**
   * @brief Access R3 velocity part.
   */
  Eigen::Map<Eigen::Vector3<Scalar>> r3_v() requires is_mutable
  {
    return Eigen::Map<Eigen::Vector3<Scalar>>(static_cast<_Derived &>(*this).data() + 0);
  }

  /**
   * @brief Const access R3 velocity part.
   */
  Eigen::Map<const Eigen::Vector3<Scalar>> r3_v() const
  {
    return Eigen::Map<const Eigen::Vector3<Scalar>>(static_cast<const _Derived &>(*this).data() + 0);
  }

  /**
   * @brief Access R3 position part.
   */
  Eigen::Map<Eigen::Vector3<Scalar>> r3_p() requires is_mutable
  {
    return Eigen::Map<Eigen::Vector3<Scalar>>(static_cast<_Derived &>(*this).data() + 3);
  }

  /**
   * @brief Const access R3 position part.
   */
  Eigen::Map<const Eigen::Vector3<Scalar>> r3_p() const
  {
    return Eigen::Map<const Eigen::Vector3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3);
  }

  /**
   * @brief Access R1 time part.
   */
  Eigen::Map<Eigen::Vector<Scalar, 1>> r1_t() requires is_mutable
  {
    return Eigen::Map<Eigen::Vector<Scalar, 1>>(static_cast<_Derived &>(*this).data() + 6);
  }

  /**
   * @brief Const access R1 time part.
   */
  Eigen::Map<const Eigen::Vector<Scalar, 1>> r1_t() const
  {
    return Eigen::Map<const Eigen::Vector<Scalar, 1>>(static_cast<const _Derived &>(*this).data() + 6);
  }

  /**
   * @brief Tranformation action on (x, y, z, t) space-time vector.
   *
   * (x, t) -> (R x + v t + p, t + tau)
   */
  template<typename EigenDerived>
  Eigen::Vector4<Scalar> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    Eigen::Vector4<Scalar> ret;
    ret.template segment<3>(0) = so3() * v.template segment<3>(0) + r3_v() * v(3) + r3_p();
    ret(3)                     = v(3) + r1_t().x();
    return ret;
  }

  /**
   * @brief Jacobian of rotation action w.r.t. group.
   *
   * \f[
   *   \mathrm{d}^r (X v)_X
   * \f]
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 4, 10> dr_action(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    Eigen::Matrix<Scalar, 4, 10> ret = Eigen::Matrix<Scalar, 4, 10>::Zero();

    // v transformation
    ret.template block<3, 3>(0, 0) = v(3) * so3().matrix();
    ret.template block<3, 3>(0, 3) = so3().matrix();
    ret.template block<3, 1>(0, 6) = r3_v();
    ret.template block<3, 3>(0, 7) = so3().dr_action(v.template segment<3>(0));

    // t transformation
    ret(3, 6) = 1;

    return ret;
  }
};

// \cond
template<typename _Scalar>
class Galilei;
// \endcond

// \cond
template<typename _Scalar>
struct liebase_info<Galilei<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = GalileiImpl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = Galilei<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of Galilei Lie group.
 *
 * @see GalileiBase for memory layout.
 */
template<typename _Scalar>
class Galilei : public GalileiBase<Galilei<_Scalar>>
{
  using Base = GalileiBase<Galilei<_Scalar>>;

  SMOOTH_GROUP_API(Galilei);

public:
  /**
   * @brief Construct from SO3 and translation.
   *
   * @param so3 orientation component.
   * @param r3_v velocity component.
   * @param r3_p position component.
   * @param r1_t time component.
   */
  template<typename SO3Derived, typename T1, typename T2>
  Galilei(
    const SO3Base<SO3Derived> & so3,
    const Eigen::MatrixBase<T1> & r3_v,
    const Eigen::MatrixBase<T2> & r3_p,
    double r1_t = 0)
  {
    Base::so3()      = static_cast<const SO3Derived &>(so3);
    Base::r3_v()     = static_cast<const T1 &>(r3_v);
    Base::r3_p()     = static_cast<const T2 &>(r3_p);
    Base::r1_t().x() = r1_t;
  }
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<Galilei<_Scalar>>> : public liebase_info<Galilei<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of Galilei Lie group.
 *
 * @see GalileiBase for memory layout.
 */
template<typename _Scalar>
class Map<Galilei<_Scalar>> : public GalileiBase<Map<Galilei<_Scalar>>>
{
  using Base = GalileiBase<Map<Galilei<_Scalar>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<const Galilei<_Scalar>>> : public liebase_info<Galilei<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of Galilei Lie group.
 *
 * @see GalileiBase for memory layout.
 */
template<typename _Scalar>
class Map<const Galilei<_Scalar>> : public GalileiBase<Map<const Galilei<_Scalar>>>
{
  using Base = GalileiBase<Map<const Galilei<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using Galileif = Galilei<float>;   ///< Galilei with float
using Galileid = Galilei<double>;  ///< Galilei with double

}  // namespace smooth
