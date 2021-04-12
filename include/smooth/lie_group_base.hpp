#ifndef SMOOTH__LIE_GROUP_BASE_HPP_
#define SMOOTH__LIE_GROUP_BASE_HPP_

#include <cstdint>

#include "storage.hpp"


namespace smooth
{

/**
 * @brief CRTP base for lie groups with common functionality and syntactic sugar
 */
template<typename Derived, uint32_t size>
class LieGroupBase;

template<
  template<typename, typename> typename _Derived,
  typename _Scalar,
  typename _Storage,
  uint32_t size
>
class LieGroupBase<_Derived<_Scalar, _Storage>, size>
{
  using Scalar = _Scalar;
  using Storage = _Storage;
  using Derived = _Derived<Scalar, Storage>;

public:
  /**
   * @brief Construct the group identity element
   */
  static Derived Identity()
  {
    Derived ret;
    ret.setIdentity();
    return ret;
  }

  /**
   * @brief Construct a random element
   *
   * @param rng a random number generator
   */
  template<typename RNG>
  static Derived Random(RNG & rng)
  {
    Derived ret;
    ret.setRandom(rng);
    return ret;
  }

  /**
   * @brief Compare two Lie group elements
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, size>
  bool isApprox(
    const _Derived<Scalar, OS> & o,
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()) const
  {
    using Tangent = std::decay_t<decltype(o.log())>;

    return (static_cast<const Derived &>(*this).inverse() * o).log().isApprox(Tangent::Zero(), eps);
  }

  /**
   * @brief Cast to different datatype
   */
  template<typename NewScalar>
  _Derived<NewScalar, DefaultStorage<NewScalar, size>> cast() const
  {
    _Derived<NewScalar, DefaultStorage<NewScalar, size>> ret;
    static_for<size>(
      [&](auto i) {
        ret.coeffs()[i] = static_cast<NewScalar>(static_cast<const Derived &>(*this).coeffs()[i]);
      });
    return ret;
  }

  /**
   * @brief Access group storage
   */
  Storage & coeffs()
  {
    return static_cast<Derived &>(*this).s_;
  }

  /**
   * @brief Const access group storage
   */
  const Storage & coeffs() const
  {
    return static_cast<const Derived &>(*this).s_;
  }

  /**
   * @brief Access raw const data pointer
   *
   * Only available for ordered storage
   */
  const Scalar * data() const requires OrderedStorageLike<Storage, Scalar, size>
  {
    return static_cast<const Derived &>(*this).coeffs().data();
  }

  /**
   * @brief Access raw data pointer
   *
   * Only available for ordered modifiable storage
   */
  Scalar * data() requires OrderedModifiableStorageLike<Storage, Scalar, size>
  {
    return static_cast<Derived &>(*this).coeffs().data();
  }

  /**
   * @brief Overload operator+
   */
  template<typename TangentDerived>
  Derived operator+(const Eigen::MatrixBase<TangentDerived> & t) const
  {
    return static_cast<const Derived &>(*this) * Derived::exp(t);
  }

  /**
   * @brief Left jacobian of the exponential
   */
  template<typename TangentDerived>
  static auto dl_exp(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (Derived::exp(t).Ad() * Derived::dr_exp(t)).eval();
  }

  /**
   * @brief Inverse of left jacobian of the exponential
   */
  template<typename TangentDerived>
  static auto dl_expinv(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (-Derived::ad(t) +  Derived::dr_expinv(t)).eval();
  }

protected:
  // protect constructor to prevent direct instantiation
  LieGroupBase() = default;
};

}  // namespace smooth

#endif  // SMOOTH__LIE_GROUP_BASE_HPP_
