#ifndef SMOOTH__SE2_HPP_
#define SMOOTH__SE2_HPP_

#include <Eigen/Core>

#include <complex>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/se2.hpp"
#include "so2.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SE3;
// \endcond

/**
 * @brief Base class for SE2 Lie group types.
 *
 * Internally represented as \f$\mathbb{U}(1) \times \mathbb{R}^2\f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [x, y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [v_x, v_y, \omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$q_z^2 + q_w^2 = 1 \f$
 * - Tangent: \f$ -\pi < \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *  q_w & -q_z & x \\
 *  q_z &  q_w & y \\
 *  0   &    0 & 1
 * \end{bmatrix}
 * \f]
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0 & -\omega_z & v_x \\
 *  \omega_z &   0 & v_y \\
 *  0 & 0 & 0
 * \end{bmatrix}
 * \f]
 */
template<typename _Derived>
class SE2Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SE2Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access SO(2) part.
   */
  Eigen::Map<SO2<Scalar>> so2() requires is_mutable
  {
    return Eigen::Map<SO2<Scalar>>(static_cast<_Derived &>(*this).data() + 2);
  }

  /**
   * @brief Const access SO(2) part.
   */
  Eigen::Map<const SO2<Scalar>> so2() const
  {
    return Eigen::Map<const SO2<Scalar>>(static_cast<const _Derived &>(*this).data() + 2);
  }

  /**
   * @brief Access T(2) part.
   */
  Eigen::Map<Eigen::Matrix<Scalar, 2, 1>> t2() requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 2, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Const access T(2) part.
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>> t2() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Tranformation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return so2() * v + t2();
  }

  /**
   * @brief Lift to SE3.
   *
   * \note SE3 header must be included.
   */
  SE3<Scalar> lift_se3() const
  {
    return SE3<Scalar>(
      so2().lift_so3(), Eigen::Matrix<Scalar, 3, 1>(t2().x(), t2().y(), Scalar(0)));
  }
};

// \cond
template<typename _Scalar>
class SE2;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<SE2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE2<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class SE2 : public SE2Base<SE2<_Scalar>>
{
  using Base = SE2Base<SE2<_Scalar>>;

  SMOOTH_GROUP_API(SE2);

public:
  /**
   * @brief Construct from SO2 and translation.
   *
   * @param so2 orientation component.
   * @param t2 translation component.
   */
  template<typename SO2Derived, typename T2Derived>
  SE2(const SO2Base<SO2Derived> & so2, const Eigen::MatrixBase<T2Derived> & t2)
  {
    Base::so2() = so2;
    Base::t2()  = t2;
  }
};

using SE2f = SE2<float>;   ///< SE2 with float
using SE2d = SE2<double>;  ///< SE2 with double

}  // namespace smooth

// MAP TYPE TRAITS

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SE2<_Scalar>>>
    : public lie_traits<smooth::SE2<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<smooth::SE2<_Scalar>> : public smooth::SE2Base<Eigen::Map<smooth::SE2<_Scalar>>>
{
  using Base = smooth::SE2Base<Eigen::Map<smooth::SE2<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SE2<_Scalar>>>
    : public lie_traits<smooth::SE2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<const smooth::SE2<_Scalar>>
    : public smooth::SE2Base<Eigen::Map<const smooth::SE2<_Scalar>>>
{
  using Base = smooth::SE2Base<Eigen::Map<const smooth::SE2<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__SE2_HPP_
