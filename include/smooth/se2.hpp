#ifndef SMOOTH__SE2_HPP_
#define SMOOTH__SE2_HPP_

#include <Eigen/Core>

#include <complex>

#include "impl/se2.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"
#include "so2.hpp"

namespace smooth {

template<typename Scalar>
class SE3;

// CRTP BASE

/**
 * @brief SE2 Lie Group represented as U(1) ⋉ R2
 *
 * Memory layout
 * =============
 * Group:    x y qz qw
 * Tangent:  vx vy Ωz
 *
 * Lie group Matrix form
 * =====================
 *
 * [ qw -qz  x ]
 * [ qz  qw  y ]
 * [  0   0  1 ]
 *
 * Lie algebra matrix form
 * =======================
 *
 * [  0 -Ωz  vx ]
 * [ Ωz   0  vy ]
 * [  0   0  1  ]
 *
 * Constraints
 * ===========
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < wz <= pi
 */
template<typename _Derived>
class SE2Base : public LieGroupBase<_Derived>
{
protected:
  using Base = LieGroupBase<_Derived>;
  SE2Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * Access SO(2) part
   */
  Eigen::Map<SO2<Scalar>> so2()
  requires is_mutable
  {
    return Eigen::Map<SO2<Scalar>>(static_cast<_Derived &>(*this).data() + 2);
  }

  /**
   * Const access SO(2) part
   */
  Eigen::Map<const SO2<Scalar>> so2() const
  {
    return Eigen::Map<const SO2<Scalar>>(static_cast<const _Derived &>(*this).data() + 2);
  }

  /**
   * Access T(2) part
   */
  Eigen::Map<Eigen::Matrix<Scalar, 2, 1>> t2()
  requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 2, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * Const access T(2) part
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>> t2() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * Tranformation action on 2D vector
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return so2() * v + t2();
  }

  /**
   * @brief Lift to SE3.
   */
  SE3<Scalar> lift_se3() const
  {
    return SE3<Scalar>(so2().lift_so3(), Eigen::Matrix<Scalar, 3, 1>(t2().x(), t2().y(), Scalar(0)));
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SE2;

template<typename _Scalar>
struct lie_traits<SE2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE2<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SE2 : public SE2Base<SE2<_Scalar>>
{
  using Base = SE2Base<SE2<_Scalar>>;

  SMOOTH_GROUP_API(SE2);

public:
  /**
   * @brief Construct from SO2 and translation
   */
  template<typename SO2Derived, typename T2Derived>
  SE2(const SO2Base<SO2Derived> & so2, const Eigen::MatrixBase<T2Derived> & t2)
  {
    Base::so2() = so2;
    Base::t2()  = t2;
  }
};

using SE2f = SE2<float>;  //! SE2 with float
using SE2d = SE2<double>;  //! SE2 with double

}  // namespace smooth

// MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SE2<_Scalar>>>
  : public lie_traits<smooth::SE2<_Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SE2<_Scalar>>
  : public smooth::SE2Base<Eigen::Map<smooth::SE2<_Scalar>>>
{
  using Base = smooth::SE2Base<Eigen::Map<smooth::SE2<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// CONST MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SE2<_Scalar>>>
  : public lie_traits<smooth::SE2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SE2<_Scalar>>
  : public smooth::SE2Base<Eigen::Map<const smooth::SE2<_Scalar>>>
{
  using Base = smooth::SE2Base<Eigen::Map<const smooth::SE2<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__SE2_HPP_
