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

#define SMOOTH_MAP_API(X)                                        \
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
                                                                 \
  Storage & coeffs() { return coeffs_; }                         \
                                                                 \
  const Storage & coeffs() const { return coeffs_; }             \
                                                                 \
private:                                                         \
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

  using Scalar     = typename traits::Scalar;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;

  template<typename NewScalar>
  using PlainObject = typename traits::template PlainObject<NewScalar>;

  // Assignment operator
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl> Derived & operator=(
    const LieGroup<OtherDerived> & o)
  {
    static_cast<Derived &>(*this).coeffs() = static_cast<const OtherDerived &>(o).coeffs();
    return static_cast<Derived &>(*this);
  }

  // Coefficient access

  // access coefficients
  auto & coeffs() { return static_cast<Derived &>(*this).coeffs(); }

  // access const coefficients
  const auto & coeffs() const { return static_cast<const Derived &>(*this).coeffs(); }

  // access coefficients
  Scalar * data() { return coeffs().data(); }

  // access const coefficients
  const Scalar * data() const { return coeffs().data(); }

  // Group API

  void setIdentity() { Impl::setIdentity(coeffs()); }

  void setRandom() { Impl::setRandom(coeffs()); }

  template<typename OtherDerived>
  PlainObject<Scalar> operator*(const LieGroup<OtherDerived> & o) const
  {
    PlainObject<Scalar> ret;
    Impl::composition(coeffs(), o.coeffs(), ret.coeffs());
    return ret;
  }

  template<typename NewScalar>
  PlainObject<NewScalar> cast() const
  {
    PlainObject<NewScalar> ret;
    ret.coeffs() = coeffs().template cast<NewScalar>();
    return ret;
  }

  PlainObject<Scalar> inverse() const
  {
    PlainObject<Scalar> ret;
    Impl::inverse(coeffs(), ret.coeffs());
    return ret;
  }

  Tangent log() const
  {
    Tangent ret;
    Impl::log(coeffs(), ret);
    return ret;
  }

  TangentMap Ad() const
  {
    TangentMap ret;
    Impl::Ad(coeffs(), ret);
    return ret;
  }

  // Tangent API

  template<typename TangentDerived>
  static PlainObject<Scalar> exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    PlainObject<Scalar> ret;
    Impl::exp(a, ret.coeffs());
    return ret;
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
