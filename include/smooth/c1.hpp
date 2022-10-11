// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <complex>

#include <Eigen/Core>

#include "detail/c1.hpp"
#include "detail/lie_group_base.hpp"
#include "detail/macro.hpp"
#include "lie_group.hpp"
#include "map.hpp"
#include "so2.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SO2;
// \endcond

/**
 * @brief Base class for C1 Lie group types.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [a, b] \f$
 * - Tangent:  \f$ \mathbf{a} = [s, \omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$ a^2 + b^2 > 0 \f$
 * - Tangent: \f$ -\pi < \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *  b & -a \\
 *  a &  b
 * \end{bmatrix} \in \mathbb{R}^{2 \times 2}
 * \f]
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   s & -\omega_z \\
 *  \omega_z &   s \\
 * \end{bmatrix} \in \mathbb{R}^{2 \times 2}
 * \f]
 */
template<typename _Derived>
class C1Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  C1Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Rotation angle.
   */
  Scalar angle() const
  {
    using std::atan2;

    return atan2(static_cast<const _Derived &>(*this).coeffs().x(), static_cast<const _Derived &>(*this).coeffs().y());
  }

  /**
   * @brief Scaling.
   */
  Scalar scaling() const
  {
    using std::sqrt;

    return sqrt(
      static_cast<const _Derived &>(*this).coeffs().x() * static_cast<const _Derived &>(*this).coeffs().x()
      + static_cast<const _Derived &>(*this).coeffs().y() * static_cast<const _Derived &>(*this).coeffs().y());
  }

  /**
   * @brief Rotation.
   */
  SO2<Scalar> so2() const
  {
    return SO2<Scalar>(c1());  // it's normalized inside SO2
  }

  /**
   * @brief Complex number representation.
   */
  std::complex<Scalar> c1() const
  {
    return std::complex<Scalar>(
      static_cast<const _Derived &>(*this).coeffs().y(), static_cast<const _Derived &>(*this).coeffs().x());
  }

  /**
   * @brief Rotation and scaling action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * v;
  }
};

// \cond
template<typename _Scalar>
class C1;
// \endcond

// \cond
template<typename _Scalar>
struct liebase_info<C1<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = C1Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = C1<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class C1 : public C1Base<C1<_Scalar>>
{
  using Base = C1Base<C1<_Scalar>>;
  SMOOTH_GROUP_API(C1);

public:
  /**
   * @brief Construct from scaling and angle.
   *
   * @param scaling strictly greater than zero.
   * @param angle angle of rotation (radians).
   */
  C1(const Scalar & scaling, const Scalar & angle)
  {
    using std::cos, std::sin;

    coeffs_.x() = scaling * sin(angle);
    coeffs_.y() = scaling * cos(angle);
  }

  /**
   * @brief Construct from complex number.
   *
   * @param c complex number.
   */
  C1(const std::complex<Scalar> & c)
  {
    coeffs_.x() = c.imag();
    coeffs_.y() = c.real();
  }
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<C1<_Scalar>>> : public liebase_info<C1<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class Map<C1<_Scalar>> : public C1Base<Map<C1<_Scalar>>>
{
  using Base = C1Base<Map<C1<_Scalar>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<const C1<_Scalar>>> : public liebase_info<C1<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class Map<const C1<_Scalar>> : public C1Base<Map<const C1<_Scalar>>>
{
  using Base = C1Base<Map<const C1<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using C1f = C1<float>;   ///< C1 with float scalar representation
using C1d = C1<double>;  ///< C1 with double scalar representation

}  // namespace smooth
