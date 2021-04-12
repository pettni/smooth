#ifndef SMOOTH__LIE_GROUP_BASE_HPP_
#define SMOOTH__LIE_GROUP_BASE_HPP_

#include <cstdint>

#include "storage.hpp"


namespace smooth
{

template<typename Derived, uint32_t size>
class LieGroupBase;

/**
 * @brief CRTP base for lie groups with common functionality and syntactic sugar
 */
template<
  template<typename, typename> typename _Derived,
  typename _Scalar,
  typename _Storage,
  uint32_t size
>
class LieGroupBase<_Derived<_Scalar, _Storage>, size>
{
  using Derived = _Derived<_Scalar, _Storage>;

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
  requires ConstStorageLike<OS, _Scalar, size>
  bool isApprox(
    const _Derived<_Scalar, OS> & o,
    const _Scalar & prec = Eigen::NumTraits<_Scalar>::dummy_precision()) const
  {
    return (static_cast<const Derived &>(*this).inverse() * o).log().norm() < 1e-5;
  }

  /**
   * @brief Cast to different datatype
   */
  template<typename NewScalar>
  _Derived<NewScalar, DefaultStorage<NewScalar, size>> cast() const
  {
    _Derived<NewScalar, DefaultStorage<NewScalar, size>> ret;
    if constexpr (is_ordered<_Storage>::value) {
      ret.coeffs() = static_cast<const Derived &>(*this).s_.template cast<NewScalar>();
    } else {
      static_for<size>(
        [&](auto i) {
          ret.coeffs()[i] = static_cast<NewScalar>(static_cast<const Derived &>(*this).coeffs()[i]);
        });
    }
    return ret;
  }

  /**
   * @brief Access group storage
   */
  _Storage & coeffs()
  {
    return static_cast<Derived &>(*this).s_;
  }

  /**
   * @brief Const access group storage
   */
  const _Storage & coeffs() const
  {
    return static_cast<const Derived &>(*this).s_;
  }

  /**
   * @brief Access raw data pointer
  */
  _Scalar * data()
  requires StorageLike<_Storage, _Scalar, size> && is_ordered_v<_Storage>
  {
    return static_cast<Derived &>(*this).s_.data();
  }

  /**
   * @brief Access raw data pointer
   */
  const _Scalar * data() const
  requires is_ordered_v<_Storage>
  {
    return static_cast<const Derived &>(*this).s_.data();
  }

protected:
  // protect constructor to prevent direct instantiation
  LieGroupBase() = default;
};

}  // namespace smooth

#endif  // SMOOTH__LIE_GROUP_BASE_HPP_
