// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "detail/macro.hpp"
#include "detail/se_k_3.hpp"
#include "detail/utils.hpp"
#include "lie_group_base.hpp"
#include "so3.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief Base class for SE_k(3) Lie group types.
 *
 * Internally represented as \f$\mathbb{S}^3 \times (\mathbb{R}^3)^k\f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [p1, ..., pk, q_x, q_y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [v1, ..., vk,, \omega_x, \omega_y, \omega_z] \f$
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
 *   \begin{bmatrix}
 *     R & P_1 & ... & P_k \\
 *     0 & 1  & ... & 0  \\
 *     \vdots
 *     0 & 0  & ... & 1  \\
 *   \end{bmatrix} \in \mathbb{R}^{3+k \times 3+k}
 * \f]
 *
 * where \f$R\f$ is a 3x3 rotation matrix and \f$ P_k = [x_k, y_k, z_k]^T \f$.
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 *   \begin{bmatrix}
 *     0        & -\omega_z & \omega_y  & u_1  & ... & u_k \\
 *    \omega_z  & 0         & -\omega_x & v_1  & ... & v_k \\
 *    -\omega_y & \omega_x  & 0         & w_1  & ... & w_k \\
 *    0         & 0         & 0         & 0    & ... & 0   \\
 *    \vdots
 *    0         & 0         & 0         & 0    & ... & 0   \\
 *   \end{bmatrix} \in \mathbb{R}^{3+k \times 3+k}
 * \f]
 */
template<typename _Derived>
class SE_K_3Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;
  using Impl = typename liebase_info<_Derived>::Impl;

protected:
  SE_K_3Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Number of R3 variables.
   */
  static constexpr auto K = Impl::K;

  /**
   * @brief Access SO(3) part.
   */
  Map<SO3<Scalar>> so3()
    requires is_mutable
  {
    return Map<SO3<Scalar>>(static_cast<_Derived &>(*this).data() + 3 * K);
  }

  /**
   * @brief Const access SO(3) part.
   */
  Map<const SO3<Scalar>> so3() const
  {
    return Map<const SO3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3 * K);
  }

  /**
   * @brief Access R3 parts.
   *
   * @tparam Ksel select part
   */
  template<int Ksel>
  Eigen::Map<Eigen::Vector3<Scalar>> r3()
    requires(is_mutable && Ksel < K)
  {
    return Eigen::Map<Eigen::Vector3<Scalar>>(static_cast<_Derived &>(*this).data() + 3 * Ksel);
  }

  /**
   * @brief Access R3 parts.
   *
   * @param k select part
   */
  Eigen::Map<Eigen::Vector3<Scalar>> r3(int k)
    requires is_mutable
  {
    assert(k < K);
    return Eigen::Map<Eigen::Vector3<Scalar>>(static_cast<_Derived &>(*this).data() + 3 * k);
  }

  /**
   * @brief Const access R3 parts.
   *
   * @tparam Ksel select part
   */
  template<int Ksel>
  Eigen::Map<const Eigen::Vector3<Scalar>> r3() const
    requires(Ksel < K)
  {
    return Eigen::Map<const Eigen::Vector3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3 * Ksel);
  }

  /**
   * @brief Const access R3 parts.
   *
   * @param k select part
   */
  Eigen::Map<const Eigen::Vector3<Scalar>> r3(int k) const
  {
    assert(k < K);
    return Eigen::Map<const Eigen::Vector3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3 * k);
  }
};

// \cond
template<typename _Scalar, int K>
class SE_K_3;
// \endcond

// \cond
template<typename _Scalar, int K>
struct liebase_info<SE_K_3<_Scalar, K>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE_K_3Impl<_Scalar, K>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE_K_3<NewScalar, K>;
};
// \endcond

/**
 * @brief Storage implementation of \f$ SE_k(3) \f$ Lie group.
 *
 * @tparam K number of translation components. Must be positive
 *
 *  - For K=1 this group is equivalent to \f$ SE(3) \f$.
 *  - For K=2 this group is a subgroup of the Galileian group.
 *
 * @see SE_K_3Base for memory layout.
 */
template<typename _Scalar, int K>
class SE_K_3 : public SE_K_3Base<SE_K_3<_Scalar, K>>
{
  using Base = SE_K_3Base<SE_K_3<_Scalar, K>>;

  SMOOTH_GROUP_API(SE_K_3);

public:
  /**
   * @brief Construct from SO3 and translation.
   *
   * @param so3 orientation component.
   * @param r3s K translation components.
   */
  template<typename SO3Derived, typename... RnDerived>
  SE_K_3(const SO3Base<SO3Derived> & so3, const Eigen::MatrixBase<RnDerived> &... r3s)
    requires(sizeof...(r3s) == K)
  {
    const auto tpl = std::forward_as_tuple(r3s...);
    Base::so3()    = static_cast<const SO3Derived &>(so3);
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-lambda-capture"
#endif
    utils::static_for<K>([this, &tpl](auto i) { Base::template r3<i>() = std::get<i>(tpl); });
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
  }
};

// \cond
template<typename _Scalar, int _K>
struct liebase_info<Map<SE_K_3<_Scalar, _K>>> : public liebase_info<SE_K_3<_Scalar, _K>>
{};
// \endcond

/**
 * @brief Memory mapping of SE_K_3 Lie group.
 *
 * @see SE_K_3Base for memory layout.
 */
template<typename _Scalar, int _K>
class Map<SE_K_3<_Scalar, _K>> : public SE_K_3Base<Map<SE_K_3<_Scalar, _K>>>
{
  using Base = SE_K_3Base<Map<SE_K_3<_Scalar, _K>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar, int _K>
struct liebase_info<Map<const SE_K_3<_Scalar, _K>>> : public liebase_info<SE_K_3<_Scalar, _K>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SE_K_3 Lie group.
 *
 * @see SE_K_3Base for memory layout.
 */
template<typename _Scalar, int _K>
class Map<const SE_K_3<_Scalar, _K>> : public SE_K_3Base<Map<const SE_K_3<_Scalar, _K>>>
{
  using Base = SE_K_3Base<Map<const SE_K_3<_Scalar, _K>>>;

  SMOOTH_CONST_MAP_API();
};

SMOOTH_END_NAMESPACE
