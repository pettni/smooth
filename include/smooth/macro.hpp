#ifndef SMOOTH__MACRO_HPP_
#define SMOOTH__MACRO_HPP_

#include "utils.hpp"

namespace smooth {

// Boilerplate for regular groups
#define SMOOTH_COMMON_TYPES(X)                                                       \
  using Scalar = _Scalar;                                                            \
  using Storage = _Storage;                                                          \
                                                                                     \
  using PlainObject = X<Scalar, DefaultStorage<Scalar, RepSize>>;                    \
                                                                                     \
  template<typename NewScalar>                                                       \
  using CastType = X<NewScalar, DefaultStorage<NewScalar, RepSize>>;                 \
                                                                                     \
  template<StorageLike NewStorage>                                                   \
  using NewStorageType = X<Scalar, NewStorage>;                                      \
                                                                                     \

// Boilerplate for regular groups and bundle
#define SMOOTH_COMMON_API(X)                                                         \
  using Tangent     = Eigen::Matrix<Scalar, Dof, 1>;                                 \
  using TangentMap  = Eigen::Matrix<Scalar, Dof, Dof>;                               \
  using Vector      = Eigen::Matrix<Scalar, ActDim, 1>;                              \
  using MatrixGroup = Eigen::Matrix<Scalar, Dim, Dim>;                               \
                                                                                     \
  X()                        = default;                                              \
  X(const X & o)             = default;                                              \
  X(X && o)                  = default;                                              \
  X & operator=(const X & o) = default;                                              \
  X & operator=(X && o)      = default;                                              \
  ~X()                       = default;                                              \
                                                                                     \
  /**                                                                                \
   * @brief Degrees of freedom at compile time                                       \
   */                                                                                \
  static constexpr Eigen::Index SizeAtCompileTime = Dof;                             \
                                                                                     \
  /**                                                                                \
   * @brief Degrees of freedom at runtime                                            \
   */                                                                                \
  Eigen::Index size() const { return Dof; };                                         \
                                                                                     \
  /**                                                                                \
   * @brief Constructor for storage                                                  \
   */                                                                                \
  template<typename S>                                                               \
  requires std::is_constructible_v<Storage, S>                                       \
  explicit X(S && s)                                                                 \
  : s_(std::forward<S>(s))                                                           \
  {}                                                                                 \
                                                                                     \
  /**                                                                                \
   * @brief Copy constructor from other storage                                      \
   */                                                                                \
  template<StorageLike OS>                                                           \
  requires ModifiableStorageLike<Storage>                                            \
  X(const NewStorageType<OS> & o)                                                    \
  {                                                                                  \
    utils::static_for<RepSize>([&](auto i) { s_[i] = o.coeffs()[i]; });              \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Copy assignment from other storage                                       \
   */                                                                                \
  template<StorageLike OS>                                                           \
  requires ModifiableStorageLike<Storage>                                            \
  X & operator=(const NewStorageType<OS> & o)                                        \
  {                                                                                  \
    utils::static_for<RepSize>([&](auto i) { s_[i] = o.s_[i]; });                    \
    return *this;                                                                    \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Construct the group identity element                                     \
   */                                                                                \
  static PlainObject Identity()                                                      \
  {                                                                                  \
    PlainObject ret;                                                                 \
    ret.setIdentity();                                                               \
    return ret;                                                                      \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Construct a random element                                               \
   *                                                                                 \
   * Set the seed with std::srand(unsigned)                                          \
   */                                                                                \
  static PlainObject Random()                                                        \
  {                                                                                  \
    PlainObject ret;                                                                 \
    ret.setRandom();                                                                 \
    return ret;                                                                      \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Compare two Lie group elements                                           \
   */                                                                                \
  template<StorageLike OS>                                                           \
  requires(std::is_same_v<Scalar, typename OS::Scalar>)                              \
  bool isApprox(                                                                     \
    const NewStorageType<OS> & o,                                                    \
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()                 \
  ) const                                                                            \
  {                                                                                  \
    double n1_sq{0}, n2_sq{0}, n12_sq{0};                                            \
    utils::static_for<RepSize>([&](auto i) {                                         \
      n1_sq += coeffs()[i] * coeffs()[i];                                            \
      n2_sq += o.coeffs()[i] * o.coeffs()[i];                                        \
      n12_sq += (coeffs()[i] - o.coeffs()[i]) * (coeffs()[i] - o.coeffs()[i]);       \
    });                                                                              \
    return std::sqrt(n12_sq) <= eps * std::min(std::sqrt(n1_sq), std::sqrt(n2_sq));  \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Cast to different scalar type                                            \
   */                                                                                \
  template<typename NewScalar>                                                       \
  CastType<NewScalar> cast() const                                                   \
  {                                                                                  \
    CastType<NewScalar> ret;                                                         \
    utils::static_for<RepSize>(                                                      \
      [&](auto i) {                                                                  \
        ret.coeffs()[i] = static_cast<NewScalar>(coeffs()[i]);                       \
      });                                                                            \
    return ret;                                                                      \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Access group storage                                                     \
   */                                                                                \
  Storage & coeffs() { return s_; }                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Const access group storage                                               \
   */                                                                                \
  const Storage & coeffs() const { return s_; }                                      \
                                                                                     \
  /**                                                                                \
   * @brief Access raw data pointer                                                  \
   */                                                                                \
  Scalar * data() requires ModifiableStorageLike<Storage>                            \
  {                                                                                  \
    return coeffs().data();                                                          \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Access raw const data pointer                                            \
   */                                                                                \
  const Scalar * data() const requires MappableStorageLike<Storage>                  \
  {                                                                                  \
    return coeffs().data();                                                          \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Overload operator*= for inplace composition                              \
   */                                                                                \
  template<StorageLike OS>                                                           \
  requires ModifiableStorageLike<Storage>                                            \
  auto & operator*=(const NewStorageType<OS> & o)                                    \
  {                                                                                  \
    *this = *this * o;                                                               \
    return *this;                                                                    \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Overload operator+ for right-plus                                        \
   *                                                                                 \
   * g + a := g1 * exp(a)                                                            \
   */                                                                                \
  template<typename TangentDerived>                                                  \
  PlainObject operator+(const Eigen::MatrixBase<TangentDerived> & t) const           \
  {                                                                                  \
    return *this * exp(t);                                                           \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Overload operator+= for inplace right-plus                               \
   *                                                                                 \
   * g + a := g1 * exp(a)                                                            \
   */                                                                                \
  template<typename TangentDerived>                                                  \
  requires ModifiableStorageLike<Storage>                                            \
  X & operator+=(const Eigen::MatrixBase<TangentDerived> & t)                        \
  {                                                                                  \
    return *this *= exp(t);                                                          \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Overload operator- for right-minus                                       \
   *                                                                                 \
   * g1 - g2 := (g2.inverse() * g1).log()                                            \
   */                                                                                \
  template<StorageLike OS>                                                           \
  Tangent operator-(const NewStorageType<OS> & o) const                              \
  {                                                                                  \
    return (o.inverse() * *this).log();                                              \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Left jacobian of the exponential                                         \
   */                                                                                \
  template<typename TangentDerived>                                                  \
  static TangentMap dl_exp(const Eigen::MatrixBase<TangentDerived> & t)              \
  {                                                                                  \
    return exp(t).Ad() * dr_exp(t);                                                  \
  }                                                                                  \
                                                                                     \
  /**                                                                                \
   * @brief Inverse of left jacobian of the exponential                              \
   */                                                                                \
  template<typename TangentDerived>                                                  \
  static TangentMap dl_expinv(const Eigen::MatrixBase<TangentDerived> & t)           \
  {                                                                                  \
    return -ad(t) + dr_expinv(t);                                                    \
  }

}  // namespace smooth

#endif  // SMOOTH__MACRO_HPP_
