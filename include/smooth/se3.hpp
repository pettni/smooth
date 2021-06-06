#ifndef SMOOTH__SE3_HPP_
#define SMOOTH__SE3_HPP_

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
requires(_Storage::Size == 7 && std::is_same_v<typename _Storage::Scalar, _Scalar>)
class SE3 : public LieGroupBase<SE3<_Scalar, _Storage>, 7>
{
private:
  _Storage s_;

public:
  // REQUIRED CONSTANTS

  static constexpr int RepSize = 7;
  static constexpr int Dof = 6;
  static constexpr int Dim = 4;
  static constexpr int ActDim = 3;

  // CONSTRUCTORS AND OPERATORS

  SMOOTH_GROUP_BOILERPLATE(SE3)

  // SE3-SPECIFIC API

  /**
   * @brief Construct from SO3 and translation
   */
  template<typename Derived>
  SE3(const Eigen::MatrixBase<Derived> & translation, const SO3<Scalar> & so3)
  requires ModifiableStorageLike<Storage>
  {
    this->translation() = translation;
    this->so3() = so3;
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
  SO3<Scalar> so3() const requires(!MappableStorageLike<Storage>)
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
  requires(!MappableStorageLike<Storage>)
  {
    return Eigen::Matrix<Scalar, 3, 1>(s_[0], s_[1], s_[2]);
  }

private:
  /**
   * @brief Compute matrix required in dr_exp and dr_expinv
   */
  template<typename Derived>
  static Eigen::Matrix<Scalar, 3, 3> calculate_q(const Eigen::MatrixBase<Derived> & a)
  {
    using std::abs, std::sqrt, std::cos, std::sin;

    const Scalar th2 = a.template tail<3>().squaredNorm();

    Scalar A, B, C;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x%29+%2F+x%5E3+at+x%3D0
      A = Scalar(1) / Scalar(6) - th2 / Scalar(120);
      // https://www.wolframalpha.com/input/?i=series+%28cos+x+-+1+%2B+x%5E2%2F2%29+%2F+x%5E4+at+x%3D0
      B = Scalar(1) / Scalar(24) - th2 / Scalar(720);
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x+-+x%5E3%2F6%29+%2F+x%5E5+at+x%3D0
      C = -Scalar(1) / Scalar(120) + th2 / Scalar(5040);
    } else {
      const Scalar th = sqrt(th2), th_4 = th2 * th2, cTh = cos(th), sTh = sin(th);
      A = (th - sTh) / (th * th2);
      B = (cTh - Scalar(1) + th2 / Scalar(2)) / th_4;
      C = (th - sTh - th * th2 / Scalar(6)) / (th_4 * th);
    }

    const Eigen::Matrix<Scalar, 3, 3> V = SO3<Scalar>::hat(a.template head<3>());
    const Eigen::Matrix<Scalar, 3, 3> W = SO3<Scalar>::hat(a.template tail<3>());

    const Scalar vdw = a.template tail<3>().dot(a.template head<3>());
    const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;

    return Scalar(0.5) * V +
           A * (WV + VW - vdw * W) +
           B * (W * WV + VW * W + vdw * (Scalar(3) * W - WW)) -
           C * Scalar(3) * vdw * WW;
  }

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    translation().setZero();
    so3().setIdentity();
  }

  /**
   * @brief Set to a random element
   */
  void setRandom() requires ModifiableStorageLike<Storage>
  {
    translation().setRandom();
    so3().setRandom();
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
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == ActDim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return so3() * x + translation();
  }

  /**
   * @brief Group composition
   */
  template<typename OS>
  PlainObject operator*(const SE3<Scalar, OS> & r) const
  {
    return PlainObject( so3() * r.translation() + translation(), so3() * r.so3());
  }

  /**
   * @brief Group inverse
   */
  PlainObject inverse() const {
    return PlainObject(-(so3().inverse() * translation()), so3().inverse());
  }

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
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  {
    return PlainObject(
      SO3<Scalar>::dl_exp(a.template tail<3>()) * a.template head<3>(),
      SO3<Scalar>::exp(a.template tail<3>())
    );
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
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
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
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
  requires(Derived::RowsAtCompileTime == Dim && Derived::ColsAtCompileTime == Dim)
  {
    Tangent ret;
    ret.template tail<3>() = SO3<Scalar>::vee(A.template topLeftCorner<3, 3>());
    ret.template head<3>() = A.template topRightCorner<3, 1>();
    return ret;
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
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
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
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
