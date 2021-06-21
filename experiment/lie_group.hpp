#ifndef LIE_GROUP_HPP_
#define LIE_GROUP_HPP_

#include <Eigen/Core>

namespace smooth {

#define SMOOTH_INHERIT_TYPEDEFS                 \
  using Base::Dof;                              \
  using Base::RepSize;                          \
  using Scalar        = typename Base::Scalar;  \
  using Tangent       = typename Base::Tangent; \
  using Base::operator=;                        \
  using Base::operator*;

#define SMOOTH_GROUP_API(X)                          \
public:                                              \
  SMOOTH_INHERIT_TYPEDEFS                            \
  X()          = default;                            \
  X(const X &) = default;                            \
  X(X &&)      = default;                            \
  X & operator=(const X &) = default;                \
  X & operator=(X &&) = default;                     \
  ~X()                = default;                     \
  template<typename OtherDerived>                    \
                                                     \
  X(const X##Base<OtherDerived> & o)                 \
  {                                                  \
    coeffs() = o.coeffs();                           \
  }                                                  \
  using Storage = Eigen::Matrix<Scalar, RepSize, 1>; \
                                                     \
  Storage & coeffs() { return coeffs_; }             \
                                                     \
  const Storage & coeffs() const { return coeffs_; } \
                                                     \
private:                                             \
  Storage coeffs_;

#define SMOOTH_MAP_API(X)                                  \
public:                                                          \
  SMOOTH_INHERIT_TYPEDEFS                                        \
  X(Scalar * p) : coeffs_(p) {}                                  \
  X(const X &) = default;                                        \
  X(X &&)      = default;                                        \
  X & operator=(const X &) = default;                            \
  X & operator=(X &&) = default;                                 \
  ~X()                = default;                                 \
                                                                 \
  using Storage = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>>; \
  Storage & coeffs() { return coeffs_; }                         \
  const Storage & coeffs() const { return coeffs_; }             \
                                                                 \
private:                                                         \
  Storage coeffs_;

#define SMOOTH_CONST_MAP_API(X)                                              \
public:                                                                \
  SMOOTH_INHERIT_TYPEDEFS                                              \
  X(Scalar * p) : coeffs_(p) {}                                        \
  X(const X &) = default;                                              \
  X(X &&)      = default;                                              \
  X & operator=(const X &) = default;                                  \
  X & operator=(X &&) = default;                                       \
  ~X()                = default;                                       \
                                                                       \
  using Storage = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>>; \
  const Storage & coeffs() const { return coeffs_; }                   \
                                                                       \
private:                                                               \
  Storage coeffs_;

template<typename T>
struct lie_traits
{};

template<typename Derived>
class LieGroup
{
protected:
  LieGroup()   = default;
  using traits = lie_traits<Derived>;
  using Impl   = typename traits::Impl;

public:
  static constexpr Eigen::Index RepSize = Impl::RepSize;
  static constexpr Eigen::Index Dof     = Impl::Dof;
  static constexpr Eigen::Index Dim     = Impl::Dim;

  using Scalar     = typename traits::Scalar;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;
  using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;

  template<typename NewScalar>
  using PlainObject = typename traits::template PlainObject<NewScalar>;

  // Assignment operator
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Derived & operator=(const LieGroup<OtherDerived> & o)
  {
    coeffs() = static_cast<const OtherDerived &>(o).coeffs();
    return static_cast<Derived &>(*this);
  }

  // Coefficient and raw data access

  /**
   * @brief Accessc coefficients
   */
  auto & coeffs() { return static_cast<Derived &>(*this).coeffs(); }

  /**
   * @brief Const accessc coefficients
   */
  const auto & coeffs() const { return static_cast<const Derived &>(*this).coeffs(); }

  /**
   * @brief Raw pointer access
   */
  Scalar * data() { return coeffs().data(); }

  /**
   * @brief Const raw pointer access
   */
  const Scalar * data() const { return coeffs().data(); }

  // Group API

  /**
   * @brief Set to identity
   */
  void setIdentity() { Impl::setIdentity(coeffs()); }

  /**
   * @brief Set to a random element
   */
  void setRandom() { Impl::setRandom(coeffs()); }

  /**
   * @brief Construct the identity element
   */
  static PlainObject<Scalar> Identity()
  {
    PlainObject<Scalar> ret;
    ret.setIdentity();
    return ret;
  }

  /**
   * @brief Construct a random element
   *
   * Set the seed with std::srand(unsigned)
   */
  static PlainObject<Scalar> Random()
  {
    PlainObject<Scalar> ret;
    ret.setRandom();
    return ret;
  }

  /**
   * @brief Return as matrix Lie group
   */
  Matrix matrix() const
  {
    Matrix ret;
    Impl::matrix(coeffs(), ret);
    return ret;
  }
  /**
   * @brief Check if (approximately) equal to other element
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  bool isApprox(
    const LieGroup<OtherDerived> & o,
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()) const
  {
    return coeffs().isApprox(o.coeffs(), eps);
  }

  /**
   * @brief Cast to different scalar type
   */
  template<typename NewScalar>
  PlainObject<NewScalar> cast() const
  {
    PlainObject<NewScalar> ret;
    ret.coeffs() = coeffs().template cast<NewScalar>();
    return ret;
  }

  /**
   * @brief Group binary operation
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  PlainObject<Scalar> operator*(const LieGroup<OtherDerived> & o) const
  {
    PlainObject<Scalar> ret;
    Impl::composition(coeffs(), o.coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Inplace group binary operation
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Derived & operator*=(const LieGroup<OtherDerived> & o)
  {
    *this = *this * o;
    return static_cast<Derived &>(*this);
  }

  /**
   * @brief Group inverse operation
   */
  PlainObject<Scalar> inverse() const
  {
    PlainObject<Scalar> ret;
    Impl::inverse(coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie group logarithm
   */
  Tangent log() const
  {
    Tangent ret;
    Impl::log(coeffs(), ret);
    return ret;
  }

  /**
   * @brief Lie group adjoint
   */
  TangentMap Ad() const
  {
    TangentMap ret;
    Impl::Ad(coeffs(), ret);
    return ret;
  }

  /**
   * @brief Right-plus
   *
   * g + a := g * exp(a)
   */
  template<typename TangentDerived>
  PlainObject<Scalar> operator+(const Eigen::MatrixBase<TangentDerived> & t) const
  {
    *this += exp(t);
    return static_cast<Derived>(*this);
  }

  /**
   * @brief Inplace right-plus
   */
  template<typename TangentDerived>
  Derived & operator+=(const Eigen::MatrixBase<TangentDerived> & t)
  {
    *this *= exp(t);
    return static_cast<Derived &>(*this);
  }

  /**
   * @brief Overload operator- for right-minus
   *
   * g1 - g2 := (g2.inverse() * g1).log()
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Tangent operator-(const LieGroup<OtherDerived> & o) const
  {
    return (o.inverse() * *this).log();
  }

  // Tangent API

  /**
   * @brief Liegroup exponential map
   */
  template<typename TangentDerived>
  static PlainObject<Scalar> exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    PlainObject<Scalar> ret;
    Impl::exp(a, ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie algebra hat
   */
  template<typename TangentDerived>
  static Matrix hat(const Eigen::MatrixBase<TangentDerived> & a)
  {
    Matrix ret;
    Impl::hat(a, ret);
    return ret;
  }

  /**
   * @brief Lie alebra vee
   */
  template<typename MatrixDerived>
  static Tangent vee(const Eigen::MatrixBase<MatrixDerived> & A)
  {
    Tangent ret;
    Impl::vee(A, ret);
    return ret;
  }

  /**
   * @brief Lie algebra adjoint
   *
   * ad_a b := [a, b]
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::ad(a, ret);
    return ret;
  }

  /**
   * @brief Lie algebra adjoint
   *
   * [a, b] := vee( hat(a) hat(b) - hat(b) hat(a) )
   */
  template<typename TangentDerived1, typename TangentDerived2>
  static Tangent lie_bracket(
      const Eigen::MatrixBase<TangentDerived1> & a1,
      const Eigen::MatrixBase<TangentDerived2> & a2
  )
  {
    return ad(a1) * a2;
  }
  /**
   * @brief Right jacobian of the exponential
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::dr_exp(a, ret);
    return ret;
  }

  /**
   * @brief Inverse of right jacobian of the exponential
   */
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::dr_expinv(a, ret);
    return ret;
  }

  /**
   * @brief Left jacobian of the exponential
   */
  template<typename TangentDerived>
  static TangentMap dl_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    return exp(a).Ad() * dr_exp(a);
  }

  /**
   * @brief Inverse of left jacobian of the exponential
   */
  template<typename TangentDerived>
  static TangentMap dl_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    return -ad(a) + dr_expinv(a);
  }
};

template<typename Stream, typename Derived>
Stream & operator<<(Stream & s, const LieGroup<Derived> & g)
{
  for (auto i = 0; i != Derived::RepSize; ++i) { s << g.coeffs()[i] << " "; }
  return s;
}

}  // namespace smooth

#endif  // LIE_GROUP_HPP_
