#ifndef SMOOTH__LIE_GROUP_BASE_HPP_
#define SMOOTH__LIE_GROUP_BASE_HPP_

#include <Eigen/Core>

namespace smooth {

template<typename T>
struct lie_traits
{};

template<typename Derived>
class LieGroupBase
{
protected:
  LieGroupBase() = default;
  using traits   = lie_traits<Derived>;
  using Impl     = typename traits::Impl;

public:
  static constexpr Eigen::Index RepSize = Impl::RepSize;
  static constexpr Eigen::Index Dof     = Impl::Dof;
  static constexpr Eigen::Index Dim     = Impl::Dim;

  static constexpr bool is_mutable = traits::is_mutable;

  using Scalar     = typename traits::Scalar;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;
  using Matrix     = Eigen::Matrix<Scalar, Dim, Dim>;

  template<typename NewScalar>
  using PlainObjectCast = typename traits::template PlainObject<NewScalar>;

  using PlainObject = PlainObjectCast<Scalar>;

  /**
   * Assignment operation from other storage type.
   */
  template<typename OtherDerived>
  requires is_mutable && std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Derived & operator=(const LieGroupBase<OtherDerived> & o)
  {
    coeffs() = static_cast<const OtherDerived &>(o).coeffs();
    return static_cast<Derived &>(*this);
  }

  // Required sizes

  /**
   * Static size (degrees of freedom).
   */
  static constexpr Eigen::Index SizeAtCompileTime = Dof;

  /**
   * Dynamic size (degrees of freedom).
   */
  Eigen::Index size() const { return Dof; }

  // Coefficient and raw data access

  /**
   * @brief Access coefficients.
   */
  auto & coeffs() requires is_mutable { return static_cast<Derived &>(*this).coeffs(); }

  /**
   * @brief Const access coefficients.
   */
  const auto & coeffs() const { return static_cast<const Derived &>(*this).coeffs(); }

  /**
   * @brief Access raw pointer.
   */
  Scalar * data() requires is_mutable { return static_cast<Derived &>(*this).coeffs().data(); }

  /**
   * @brief Const access raw pointer.
   */
  const Scalar * data() const { return static_cast<const Derived &>(*this).coeffs().data(); }

  // Group API

  /**
   * @brief Set to group identity element.
   */
  void setIdentity() requires is_mutable { Impl::setIdentity(coeffs()); }

  /**
   * @brief Set to a random element.
   *
   * Set the seed with std::srand(unsigned).
   */
  void setRandom() requires is_mutable { Impl::setRandom(coeffs()); }

  /**
   * @brief Construct the identity element.
   */
  static PlainObject Identity()
  {
    PlainObject ret;
    ret.setIdentity();
    return ret;
  }

  /**
   * @brief Construct a random element.
   *
   * Set the seed with std::srand(unsigned).
   */
  static PlainObject Random()
  {
    PlainObject ret;
    ret.setRandom();
    return ret;
  }

  /**
   * @brief Return as matrix Lie group element.
   */
  Matrix matrix() const
  {
    Matrix ret;
    Impl::matrix(coeffs(), ret);
    return ret;
  }

  /**
   * @brief Check if (approximately) equal to other element `o`.
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  bool isApprox(
    const LieGroupBase<OtherDerived> & o,
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()) const
  {
    return coeffs().isApprox(o.coeffs(), eps);
  }

  /**
   * @brief Cast to different scalar type.
   */
  template<typename NewScalar>
  PlainObjectCast<NewScalar> cast() const
  {
    PlainObjectCast<NewScalar> ret;
    ret.coeffs() = coeffs().template cast<NewScalar>();
    return ret;
  }

  /**
   * @brief Group binary composition operation.
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  PlainObject operator*(const LieGroupBase<OtherDerived> & o) const
  {
    PlainObject ret;
    Impl::composition(coeffs(), o.coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Inplace group binary composition operation.
   */
  template<typename OtherDerived>
  requires is_mutable && std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Derived & operator*=(const LieGroupBase<OtherDerived> & o)
  {
    coeffs() = (*this * o).coeffs();
    return static_cast<Derived &>(*this);
  }

  /**
   * @brief Group inverse operation.
   */
  PlainObject inverse() const
  {
    PlainObject ret;
    Impl::inverse(coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie group logarithm.
   */
  Tangent log() const
  {
    Tangent ret;
    Impl::log(coeffs(), ret);
    return ret;
  }

  /**
   * @brief Lie group adjoint.
   *
   * Ad_X a := ( X hat(a) X.inverse() ).vee()
   */
  TangentMap Ad() const
  {
    TangentMap ret;
    Impl::Ad(coeffs(), ret);
    return ret;
  }

  /**
   * @brief Right-plus.
   *
   * g + a := g * exp(a)
   */
  template<typename TangentDerived>
  PlainObject operator+(const Eigen::MatrixBase<TangentDerived> & t) const
  {
    return *this * exp(t);
  }

  /**
   * @brief Inplace right-plus.
   */
  template<typename TangentDerived>
  requires is_mutable
  Derived & operator+=(const Eigen::MatrixBase<TangentDerived> & t)
  {
    *this *= exp(t);
    return static_cast<Derived &>(*this);
  }

  /**
   * @brief Right-minus.
   *
   * g1 - g2 := (g2.inverse() * g1).log()
   */
  template<typename OtherDerived>
  requires std::is_same_v<Impl, typename lie_traits<OtherDerived>::Impl>
  Tangent operator-(const LieGroupBase<OtherDerived> & o) const
  {
    return (o.inverse() * *this).log();
  }

  // Tangent API

  /**
   * @brief Lie group exponential map.
   */
  template<typename TangentDerived>
  static PlainObject exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    PlainObject ret;
    Impl::exp(a, ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie algebra hat.
   *
   * Maps a Dofx1 Rn parameterization of a matrix Lie algebra element
   * to the corresponding matrix Lie algebra element.
   */
  template<typename TangentDerived>
  static Matrix hat(const Eigen::MatrixBase<TangentDerived> & a)
  {
    Matrix ret;
    Impl::hat(a, ret);
    return ret;
  }

  /**
   * @brief Lie alebra vee.
   *
   * Maps a matrix Lie algebra element to its Rn parameterization.
   */
  template<typename MatrixDerived>
  static Tangent vee(const Eigen::MatrixBase<MatrixDerived> & A)
  {
    Tangent ret;
    Impl::vee(A, ret);
    return ret;
  }

  /**
   * @brief Lie algebra adjoint.
   *
   * ad_a b := [a, b].
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::ad(a, ret);
    return ret;
  }

  /**
   * @brief Lie algebra bracket.
   *
   * [a, b] := vee( hat(a) hat(b) - hat(b) hat(a) ).
   */
  template<typename TangentDerived1, typename TangentDerived2>
  static Tangent lie_bracket(
    const Eigen::MatrixBase<TangentDerived1> & a1, const Eigen::MatrixBase<TangentDerived2> & a2)
  {
    return ad(a1) * a2;
  }
  /**
   * @brief Right jacobian of the exponential map.
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::dr_exp(a, ret);
    return ret;
  }

  /**
   * @brief Inverse of right jacobian of the exponential map.
   */
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    Impl::dr_expinv(a, ret);
    return ret;
  }

  /**
   * @brief Left jacobian of the exponential map.
   */
  template<typename TangentDerived>
  static TangentMap dl_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    return exp(a).Ad() * dr_exp(a);
  }

  /**
   * @brief Inverse of left jacobian of the exponential map.
   */
  template<typename TangentDerived>
  static TangentMap dl_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    return -ad(a) + dr_expinv(a);
  }
};


/**
 * @brief Stream operator for Lie groups that ouputs the coefficients.
 */
template<typename Stream, typename Derived>
Stream & operator<<(Stream & s, const LieGroupBase<Derived> & g)
{
  for (auto i = 0; i != Derived::RepSize; ++i) { s << g.coeffs()[i] << " "; }
  return s;
}

}  // namespace smooth

#endif  // SMOOTH__LIE_GROUP_BASE_HPP_
