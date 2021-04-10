#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>

#include "concepts.hpp"
#include "common.hpp"


namespace smooth
{

template<typename _Scalar, typename _S = DefaultStorage<_Scalar, 4>>
requires ConstStorageLike<_S, _Scalar, 4>
class SO3
{
public:
  static constexpr uint32_t size = 4;
  static constexpr uint32_t dof = 3;
  static constexpr uint32_t dim = 3;

  /////////////////
  // GENERIC API //
  /////////////////

  using Storage = _S;
  using Scalar = _Scalar;

  using Group = SO3<Scalar, DefaultStorage<Scalar, size>>;
  using Tangent = Eigen::Matrix<Scalar, dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, dof, dof>;
  using Algebra = Eigen::Matrix<Scalar, dim, dim>;

  SO3() = default;
  SO3(const SO3 & o) = default;
  SO3(SO3 && o) = default;
  SO3 & operator=(const SO3 & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }
  SO3 & operator=(SO3 && o) = default;

  /**
   * @brief Copy constructor from other SO3
   */
  template<typename OS>
  requires ConstStorageLike<OS, Scalar, size>
  SO3(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
  }

  /**
   * @brief Forwarding constructor to storage for map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, T *>
  explicit SO3(T * ptr)
  : s_(ptr) {}

  /**
   * @brief Forwarding constructor to storage for const map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, const T *>
  explicit SO3(const T * ptr)
  : s_(ptr) {}

  /**
   * @brief Copy assignment from other SO3
   */
  template<typename OS>
  requires ConstStorageLike<OS, Scalar, size>
  SO3 & operator=(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  /**
   * @brief Access group storage
   */
  Storage & storage() requires StorageLike<Storage, Scalar, 4>
  {
    return s_;
  }

  /**
   * @brief Const access group storage
   */
  const Storage & storage() const
  {
    return s_;
  }

  /**
   * @brief Access raw data pointer
   */
  Scalar & data() requires StorageLike<Storage, Scalar, 4>
  {
    return s_.data();
  }

  /**
   * @brief Access raw data pointer
   */
  const Scalar & data() const
  {
    return s_.data();
  }

private:
  Storage s_;

  template<typename OtherScalar, typename OS>
  requires std::is_same_v<Scalar, OtherScalar>
  friend class SO3;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

////////////////////
// SPECIFIC STUFF //
////////////////////

public:
  /**
   * @brief Construct from quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & q)
  requires std::is_same_v<typename Derived::Scalar, Scalar>
  {
    s_[0] = q.x();
    s_[1] = q.y();
    s_[2] = q.z();
    s_[3] = q.w();
    normalize();
  }

  // REQUIRED API

  /**
   * @brief Set to identity element
   */
  void setIdentity()
  {
    s_[0] = Scalar(1);
    s_[1] = Scalar(0);
    s_[2] = Scalar(0);
    s_[3] = Scalar(0);
  }

  /**
   * @brief Group logarithm
   */
  Tangent log() const
  {
    return Tangent::Zero();
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    Eigen::Quaternion<Scalar> q(s_[3], s_[0], s_[1], s_[2]);
    return q.toRotationMatrix();
  }

  /**
   * @brief Group exponential
   */
  static Group exp(const Tangent & t)
  {
    return Group();
  }

  /**
   * @brief Algebra adjoint
   */
  static TangentMap ad(const Tangent & t)
  {
    return hat(t);
  }

  /**
   * @brief Algebra hat
   */
  static Algebra hat(const Tangent & t)
  {
    return (Algebra() <<
           Scalar(0), -t.z(), t.y(),
           t.z(), Scalar(0), -t.x(),
           -t.y(), t.x(), Scalar(0)
    ).finished();
  }

  // EXTRA API

  void normalize()
  {
    using std::sqrt;
    const Scalar norm_sq = s_[0] * s_[0] + s_[1] * s_[1] + s_[2] * s_[2] + s_[3] * s_[3];

    const Scalar norm = sqrt(norm_sq);
    s_[0] /= norm;
    s_[1] /= norm;
    s_[2] /= norm;
    s_[3] /= norm;
  }
};


/**
 * @brief Generic map type
 */
template<LieGroupLike G>
using Map = change_template_args_t<
  G,
  typename G::Scalar,
  Eigen::Map<DefaultStorage<typename G::Scalar, G::size>>
>;

/**
 * @brief Generic const map type
 */
template<LieGroupLike G>
using ConstMap = change_template_args_t<
  G,
  typename G::Scalar,
  const Eigen::Map<const DefaultStorage<typename G::Scalar, G::size>>
>;


// Group typedefs
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_
