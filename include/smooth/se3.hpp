#ifndef SMOOTH__SE3_HPP_
#define SMOOTH__SE3_HPP_

#include <random>

#include "common.hpp"
#include "concepts.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"
#include "so3.hpp"

namespace smooth
{
/**
 * @brief SE3 Lie Group
 *
 * Memory layout
 * =============
 * Group:    x y z qx qy qz qw
 * Tangent:  vx vy vz Ωx Ωy Ωz
 *
 * Constraints
 * ===========
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx Ωy Ωz <= pi, 0 <= Ωw <= pi
 */
template<typename _Scalar, StorageLike _Storage = DefaultStorage<_Scalar, 7>>
requires(_Storage::SizeAtCompileTime == 7 && std::is_same_v<typename _Storage::Scalar, _Scalar>)
class SE3 : public LieGroupBase<SE3<_Scalar, _Storage>, 7>
{
private:
  _Storage s_;

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t lie_size = 7;
  static constexpr uint32_t lie_dof = 6;
  static constexpr uint32_t lie_dim = 4;
  static constexpr uint32_t lie_actdim = 3;

  // CONSTRUCTORS AND OPERATORS

  SMOOTH_GROUP_BOILERPLATE(SE3)

  // SE3-SPECIFIC API

  /**
   * @brief Construct from SO3 and translation
   */
  template<typename Derived>
  SE3(const SO3<Scalar> & so3, const Eigen::MatrixBase<Derived> & translation)
  requires ModifiableStorageLike<Storage>
  {
    s_[0] = translation(0);
    s_[1] = translation(1);
    s_[2] = translation(2);
    s_[3] = so3.coeffs()[0];
    s_[4] = so3.coeffs()[1];
    s_[5] = so3.coeffs()[2];
    s_[6] = so3.coeffs()[3];
  }

  /**
   * @brief Access const SO3 part
   */
  Map<const SO3<Scalar>> so3() const requires MappableStorageLike<Storage>
  {
    return Map<const SO3<Scalar>>(s_.data() + 3);
  }

  /**
   * @brief Access SO2 part
   */
  Map<SO3<Scalar>> so3() requires ModifiableStorageLike<Storage>
  {
    return Map<SO3<Scalar>>(s_.data() + 3);
  }

  /**
   * @brief Access SO2 part by copy
   */
  SO3<Scalar> so3() const requires (!MappableStorageLike<Storage>)
  {
    return SO3<Scalar>(Eigen::Quaternion<Scalar>(s_[6], s_[3], s_[4], s_[5]));
  }

  /**
   * @brief Access const E3 part
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> translation() const
  requires MappableStorageLike<Storage>
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(s_.data());
  }

  /**
   * @brief Access E3 part
   */
  Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> translation()
  requires ModifiableStorageLike<Storage>
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 3, 1>>(s_.data());
  }

  /**
   * @brief Access E3 part by copy
   */
  Eigen::Matrix<Scalar, 3, 1> translation() const
  requires (!MappableStorageLike<Storage>)
  {
    return Eigen::Matrix<Scalar, 3, 1>(s_[0], s_[1], s_[2]);
  }

private:
  template<typename Derived>
  static Eigen::Matrix<Scalar, 3, 3> calculate_q(const Eigen::MatrixBase<Derived> & a)
  {
    using std::abs, std::sqrt, std::cos, std::sin;

    const Scalar th_sq = a.template tail<3>().squaredNorm();
    const Scalar th = sqrt(th_sq);

    Eigen::Matrix<Scalar, 3, 3> V = SO3<Scalar>::hat(a.template head<3>());
    Eigen::Matrix<Scalar, 3, 3> W = SO3<Scalar>::hat(a.template tail<3>());

    Eigen::Matrix<Scalar, 3, 3> Q = Scalar(0.5) * V;

    if (th < Scalar(eps)) {
      // small angle approx
      Q += Scalar(1) / Scalar(6) * (W * V + V * W);
    } else {
      // pre-calc some quantities that are used multiple times
      const Scalar th_4 = th_sq * th_sq;
      const Scalar vdw = a.template tail<3>().dot(a.template head<3>());
      const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;
      const Scalar cTh = cos(th);
      const Scalar sTh = sin(th);

      Q += (th - sTh) / (th * th_sq) * (WV + VW - vdw * W);
      Q += (cTh - Scalar(1) + th_sq / Scalar(2)) / th_4 *
        (W * WV + VW * W + vdw * (Scalar(3) * W - WW));
      Q -= Scalar(3) * vdw * (th - sTh - th * th_sq / Scalar(6)) / (th_4 * th) * WW;
    }
    return Q;
  }

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    s_[0] = Scalar(0); s_[1] = Scalar(0); s_[2] = Scalar(0); s_[3] = Scalar(0);
    s_[4] = Scalar(0); s_[5] = Scalar(0); s_[6] = Scalar(1);
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng) requires ModifiableStorageLike<Storage>&& std::is_floating_point_v<Scalar>
  {
    const Scalar x = filler<Scalar>(rng, 0);
    const Scalar y = filler<Scalar>(rng, 0);
    const Scalar z = filler<Scalar>(rng, 0);

    SO3<Scalar> so3;
    so3.setRandom(rng);

    // x y qz qw
    s_[0] = x; s_[1] = y; s_[2] = z;
    s_[3] = so3.coeffs()[0]; s_[4] = so3.coeffs()[1];
    s_[5] = so3.coeffs()[2]; s_[6] = so3.coeffs()[3];
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    MatrixGroup ret;
    ret.setIdentity();
    ret.template topLeftCorner<3, 3>() = so3().matrix_group();
    ret.template topRightCorner<3, 1>() = translation();
    return ret;
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_actdim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return so3() * x + translation();
  }

  /**
   * @brief Group composition
   */
  template<typename OS>
  Group operator*(const SE3<Scalar, OS> & r) const
  {
    return Group(so3() * r.so3(), so3() * r.translation() + translation());
  }

  /**
   * @brief Group inverse
   */
  Group inverse() const {return Group(so3().inverse(), -(so3().inverse() * translation()));}

  /**
   * @brief Group logarithm
   */
  Tangent log() const
  {
    Tangent ret;
    ret.template tail<3>() = so3().log();
    ret.template head<3>() = SO3<Scalar>::dl_expinv(ret.template tail<3>()) * translation();
    return ret;
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    TangentMap ret;
    ret.template topLeftCorner<3, 3>() = so3().matrix_group();
    ret.template topRightCorner<3, 3>() =
      SO3<Scalar>::hat(translation()) * ret.template topLeftCorner<3, 3>();
    ret.template bottomRightCorner<3, 3>() = ret.template topLeftCorner<3, 3>();
    ret.template bottomLeftCorner<3, 3>().setZero();
    return ret;
  }

  // REQUIRED TANGENT API

  /**
   * @brief Group exponential
   */
  template<typename Derived>
  static Group exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return Group(
      SO3<Scalar>::exp(a.template tail<3>()),
      SO3<Scalar>::dl_exp(a.template tail<3>()) * a.template head<3>());
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.template topLeftCorner<3, 3>() = SO3<Scalar>::hat(a.template tail<3>());
    ret.template topRightCorner<3, 3>() = SO3<Scalar>::hat(a.template head<3>());
    ret.template bottomRightCorner<3, 3>() = ret.template topLeftCorner<3, 3>();
    ret.template bottomLeftCorner<3, 3>().setZero();
    return ret;
  }

  /**
   * @brief Algebra hat
   */
  template<typename Derived>
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    MatrixGroup ret;
    ret.setZero();
    ret.template topLeftCorner<3, 3>() = SO3<Scalar>::hat(a.template tail<3>());
    ret.template topRightCorner<3, 1>() = a.template head<3>();
    return ret;
  }

  /**
   * @brief Algebra vee
   */
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  requires(Derived::RowsAtCompileTime == lie_dim && Derived::ColsAtCompileTime == lie_dim)
  {
    Tangent a;
    a.template tail<3>() = SO3<Scalar>::vee(A.template topLeftCorner<3, 3>());
    a.template head<3>() = A.template topRightCorner<3, 1>();
    return a;
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.template topLeftCorner<3, 3>() = SO3<Scalar>::dr_exp(a.template tail<3>());
    ret.template topRightCorner<3, 3>() = calculate_q(-a);
    ret.template bottomRightCorner<3, 3>() = ret.template topLeftCorner<3, 3>();
    ret.template bottomLeftCorner<3, 3>().setZero();
    return ret;
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.template topLeftCorner<3, 3>() = SO3<Scalar>::dr_expinv(a.template tail<3>());
    ret.template topRightCorner<3, 3>() =
      -ret.template topLeftCorner<3, 3>() * calculate_q(-a) * ret.template topLeftCorner<3, 3>();
    ret.template bottomRightCorner<3, 3>() = ret.template topLeftCorner<3, 3>();
    ret.template bottomLeftCorner<3, 3>().setZero();
    return ret;
  }
};

using SE3f = SE3<float>;
using SE3d = SE3<double>;

}  // namespace smooth

#endif  // SMOOTH__SE3_HPP_
