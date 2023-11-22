// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "detail/macro.hpp"
#include "detail/se2.hpp"
#include "lie_group_base.hpp"
#include "so2.hpp"

SMOOTH_BEGIN_NAMESPACE

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
 * \end{bmatrix} \in \mathbb{R}^{3 \times 3}
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
 * \end{bmatrix} \in \mathbb{R}^{3 \times 3}
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
  Map<SO2<Scalar>> so2()
    requires is_mutable
  {
    return Map<SO2<Scalar>>(static_cast<_Derived &>(*this).data() + 2);
  }

  /**
   * @brief Const access SO(2) part.
   */
  Map<const SO2<Scalar>> so2() const { return Map<const SO2<Scalar>>(static_cast<const _Derived &>(*this).data() + 2); }

  /**
   * @brief Access R2 part.
   */
  Eigen::Map<Eigen::Vector2<Scalar>> r2()
    requires is_mutable
  {
    return Eigen::Map<Eigen::Vector2<Scalar>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Const access R2 part.
   */
  Eigen::Map<const Eigen::Vector2<Scalar>> r2() const
  {
    return Eigen::Map<const Eigen::Vector2<Scalar>>(static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Return as 2D Eigen transform.
   */
  Eigen::Transform<Scalar, 2, Eigen::Isometry> isometry() const
  {
    return Eigen::Translation<Scalar, 2>(r2()) * Eigen::Rotation2D<Scalar>(so2().angle());
  }

  /**
   * @brief Tranformation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Vector2<Scalar> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return so2() * v + r2();
  }

  /**
   * @brief Jacobian of rotation action w.r.t. group.
   *
   * \f[
   *   \mathrm{d}^r (X v)_X
   * \f]
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 3> dr_action(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    Eigen::Matrix<Scalar, 2, 3> ret;
    ret.template leftCols<2>()  = so2().matrix();
    ret.template rightCols<1>() = so2().dr_action(v);
    return ret;
  }

  /**
   * @brief Lift to SE3.
   *
   * @note SE3 header must be included.
   */
  SE3<Scalar> lift_se3() const
  {
    return SE3<Scalar>(so2().lift_so3(), Eigen::Vector3<Scalar>(r2().x(), r2().y(), Scalar(0)));
  }
};

// \cond
template<typename _Scalar>
class SE2;
// \endcond

// \cond
template<typename _Scalar>
struct liebase_info<SE2<_Scalar>>
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
   * @brief Construct from SO2 and R2.
   *
   * @param so2 orientation component.
   * @param r2 translation component.
   */
  template<typename SO2Derived, typename T2Derived>
  SE2(const SO2Base<SO2Derived> & so2, const Eigen::MatrixBase<T2Derived> & r2)
  {
    Base::so2() = static_cast<const SO2Derived &>(so2);
    Base::r2()  = static_cast<const T2Derived &>(r2);
  }

  /**
   * @brief Construct from Eigen transform.
   */
  explicit SE2(const Eigen::Transform<Scalar, 2, Eigen::Isometry> & t)
  {
    Eigen::Matrix2<Scalar> rotmat = t.rotation();
    coeffs().x()                  = t.translation().x();
    coeffs().y()                  = t.translation().y();
    coeffs().z()                  = rotmat(1, 0);  // sin(angle)
    coeffs().w()                  = rotmat(0, 0);  // cos(angle)
  }
};

// MAP TYPE TRAITS

// \cond
template<typename _Scalar>
struct liebase_info<Map<SE2<_Scalar>>> : public liebase_info<SE2<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class Map<SE2<_Scalar>> : public SE2Base<Map<SE2<_Scalar>>>
{
  using Base = SE2Base<Map<SE2<_Scalar>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<const SE2<_Scalar>>> : public liebase_info<SE2<_Scalar>>
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
class Map<const SE2<_Scalar>> : public SE2Base<Map<const SE2<_Scalar>>>
{
  using Base = SE2Base<Map<const SE2<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using SE2f = SE2<float>;   ///< SE2 with float
using SE2d = SE2<double>;  ///< SE2 with double

SMOOTH_END_NAMESPACE

#if __has_include(<format>)
#include <format>
#include <string>

template<class Scalar>
struct std::formatter<smooth::SE2<Scalar>>
{
  std::string m_format;

  constexpr auto parse(std::format_parse_context & ctx)
  {
    m_format = "{:";
    for (auto it = ctx.begin(); it != ctx.end(); ++it) {
      char c = *it;
      m_format += c;
      if (c == '}') return it;
    }
    return ctx.end();
  }

  auto format(const smooth::SE2<Scalar> & obj, std::format_context & ctx) const
  {
    const auto fmtSting = std::format("r2: [{0}, {0}], so2: {0}", m_format);
    return std::vformat_to(ctx.out(), fmtSting, std::make_format_args(obj.r2().x(), obj.r2().y(), obj.so2().angle()));
  }
};

#endif
