#ifndef SMOOTH__SO2_HPP_
#define SMOOTH__SO2_HPP_

#include <Eigen/Core>

#include <complex>

#include "impl/so2.hpp"
#include "internal/macro.hpp"
#include "lie_group_base.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SO3;
// \endcond

/**
 * @brief Base class for SO2 Lie group types.
 *
 * Internally represented as \f$\mathbb{U}(1)\f$ (complex numbers).
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [\omega_z] \f$
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
 *  q_w & -q_z \\
 *  q_z &  q_w 
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
 *   0 & -\omega_z \\
 *  \omega_z &   0 \\
 * \end{bmatrix}
 * \f]
 */
template<typename _Derived>
class SO2Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SO2Base()  = default;

public:

  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Angle represetation.
   */
  Scalar angle() const { return Base::log().x(); }

  /**
   * @brief Complex number (U(1)) representation.
   */
  std::complex<Scalar> u1() const
  {
    return std::complex<Scalar>(Base::coeffs().y(), Base::coeffs().x());
  }

  /**
   * @brief Rotation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * v;
  }

  /**
   * @brief Lift to SO3.
   *
   * Rotation of SO2 is embedded in SO3 as a rotation around the z axis.
   *
   * \note SO3 header must be included.
   */
  SO3<Scalar> lift_so3() const
  {
    using std::cos, std::sin;

    const Scalar yaw = Base::log().x();
    return SO3<Scalar>(Eigen::Quaternion<Scalar>(cos(yaw / 2), 0, 0, sin(yaw / 2)));
  }
};

// \cond
template<typename _Scalar>
class SO2;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<SO2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SO2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO2<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SO2 Lie group.
 * 
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class SO2 : public SO2Base<SO2<_Scalar>>
{
  using Base = SO2Base<SO2<_Scalar>>;
  SMOOTH_GROUP_API(SO2);

public:
  /**
   * @brief Construct from coefficients.
   *
   * @param qz sine of rotation angle
   * @param qw cosine of rotation angle
   *
   * @note Inputs are are normalized to ensure group constraint.
   */
  SO2(const Scalar & qz, const Scalar & qw)
  {
    using std::sqrt;

    const Scalar n = sqrt(qw * qw + qz * qz);
    coeffs_.x() = qz / n;
    coeffs_.y() = qw / n;
  }

  /**
   * @brief Construct from angle.
   *
   * @param angle angle of rotation (radians).
   */
  explicit SO2(const Scalar & angle)
  {
    using std::cos, std::sin;

    coeffs_.x() = sin(angle);
    coeffs_.y() = cos(angle);
  }

  /**
   * @brief Construct from complex number.
   *
   * @param c complex number.
   *
   * @note Input is normalized to ensure group constraint.
   */
  SO2(const std::complex<Scalar> & c)
  {
    using std::sqrt;

    const Scalar n = sqrt(c.imag() * c.imag() + c.real() * c.real());
    coeffs_.x() = c.imag() / n;
    coeffs_.y() = c.real() / n;
  }
};

using SO2f = SO2<float>;  ///< SO2 with float scalar representation
using SO2d = SO2<double>;  ///< SO2 with double scalar representation

}  // namespace smooth

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO2<_Scalar>>>
    : public lie_traits<smooth::SO2<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SO2 Lie group.
 * 
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<smooth::SO2<_Scalar>> : public smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>
{
  using Base = smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SO2<_Scalar>>>
    : public lie_traits<smooth::SO2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SO2 Lie group.
 * 
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<const smooth::SO2<_Scalar>>
    : public smooth::SO2Base<Eigen::Map<const smooth::SO2<_Scalar>>>
{
  using Base = smooth::SO2Base<Eigen::Map<const smooth::SO2<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__SO2_HPP_
