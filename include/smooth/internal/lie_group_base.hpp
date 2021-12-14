// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__INTERNAL__LIE_GROUP_BASE_HPP_
#define SMOOTH__INTERNAL__LIE_GROUP_BASE_HPP_

#include <Eigen/Core>

namespace smooth {

/**
 * @brief Type trait that maps a type to Lie group operations.
 *
 * Required members:
 *  - `typename Impl`: Lie group operations
 *  - `typename Scalar`: Scalar type (\p float, \p double, ...)
 *  - `typename template<NewScalar> PlainObject`: Default return type
 */
template<typename T>
struct liebase_info
{};

/**
 * @brief Base class for Lie group types
 */
template<typename Derived>
class LieGroupBase
{
  Derived & derived() noexcept { return static_cast<Derived &>(*this); }

  const Derived & cderived() const noexcept { return static_cast<const Derived &>(*this); }

protected:
  LieGroupBase() = default;

  //! CRTP traits
  using traits = liebase_info<Derived>;

  //! Group-specific Lie group implementation
  using Impl = typename traits::Impl;

  //! True if underlying storage supports modification
  static constexpr bool is_mutable = traits::is_mutable;

public:
  //! Number of scalars in internal representation.
  static constexpr Eigen::Index RepSize = Impl::RepSize;
  //! Degrees of freedom of manifold (equal to tangent space dimension).
  static constexpr Eigen::Index Dof = Impl::Dof;
  //! Side of Lie group matrix representation.
  static constexpr Eigen::Index Dim = Impl::Dim;

  //! Scalar type
  using Scalar = typename traits::Scalar;
  //! Lie group matrix type
  using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;
  //! Lie group parameterized tangent type
  using Tangent = Eigen::Matrix<Scalar, Dof, 1>;
  //! Matrix representing map between tangent elements
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;
  //! Plain return type with different scalar
  template<typename NewScalar>
  using CastT = typename traits::template PlainObject<NewScalar>;
  //! Plain return type
  using PlainObject = CastT<Scalar>;

  /**
   * Assignment operation from other storage type.
   */
  template<typename OtherDerived>
    requires(is_mutable && std::is_same_v<Impl, typename liebase_info<OtherDerived>::Impl>)
  Derived & operator=(const LieGroupBase<OtherDerived> & o) noexcept
  {
    derived().coeffs() = static_cast<const OtherDerived &>(o).coeffs();
    return derived();
  }

  // Group API

  /**
   * @brief Dynamic size (degrees of freedom).
   */
  Eigen::Index dof() const noexcept { return Dof; }

  /**
   * @brief Set to group identity element.
   */
  void setIdentity() noexcept { Impl::setIdentity(derived().coeffs()); }

  /**
   * @brief Set to a random element.
   *
   * Set the seed with \p std::srand(unsigned).
   */
  void setRandom() noexcept { Impl::setRandom(derived().coeffs()); }

  /**
   * @brief Construct the identity element.
   */
  [[nodiscard]] static PlainObject Identity() noexcept
  {
    PlainObject ret;
    ret.setIdentity();
    return ret;
  }

  /**
   * @brief Construct a random element.
   *
   * Set the seed with \p std::srand(unsigned).
   */
  [[nodiscard]] static PlainObject Random() noexcept
  {
    PlainObject ret;
    ret.setRandom();
    return ret;
  }

  /**
   * @brief Return as matrix Lie group element in \f$ \mathbb{R}^{\mathtt{dim} \times \mathtt{dim}}
   * \f$.
   */
  Matrix matrix() const noexcept
  {
    Matrix ret;
    Impl::matrix(cderived().coeffs(), ret);
    return ret;
  }

  /**
   * @brief Check if (approximately) equal to other element `o`.
   */
  template<typename OtherDerived>
    requires(std::is_same_v<Impl, typename liebase_info<OtherDerived>::Impl>)
  bool isApprox(
    const LieGroupBase<OtherDerived> & o,
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()) const noexcept
  {
    return cderived().coeffs().isApprox(static_cast<const OtherDerived &>(o).coeffs(), eps);
  }

  /**
   * @brief Cast to different scalar type.
   */
  template<typename NewScalar>
  [[nodiscard]] CastT<NewScalar> cast() const noexcept
  {
    CastT<NewScalar> ret;
    ret.coeffs() = cderived().coeffs().template cast<NewScalar>();
    return ret;
  }

  /**
   * @brief Group binary composition operation.
   */
  template<typename OtherDerived>
    requires(std::is_same_v<Impl, typename liebase_info<OtherDerived>::Impl>)
  PlainObject operator*(const LieGroupBase<OtherDerived> & o) const noexcept
  {
    PlainObject ret;
    Impl::composition(
      cderived().coeffs(), static_cast<const OtherDerived &>(o).coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Inplace group binary composition operation.
   */
  template<typename OtherDerived>
    requires(is_mutable && std::is_same_v<Impl, typename liebase_info<OtherDerived>::Impl>)
  Derived & operator*=(const LieGroupBase<OtherDerived> & o) noexcept
  {
    derived().coeffs() = (*this * o).coeffs();
    return derived();
  }

  /**
   * @brief Group inverse operation.
   */
  [[nodiscard]] PlainObject inverse() const noexcept
  {
    PlainObject ret;
    Impl::inverse(cderived().coeffs(), ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie group logarithm.
   *
   * @return tangent logarithm element.
   */
  [[nodiscard]] Tangent log() const noexcept
  {
    Tangent ret;
    Impl::log(cderived().coeffs(), ret);
    return ret;
  }

  /**
   * @brief Lie group adjoint.
   *
   * @return Matrix \f$ \mathbf{Ad}_\mathbf{X} \f$ s.t.  \f$ \mathbf{Ad_X} \mathbf{a} = ( \mathbf{X}
   * \mathbf{a}^\wedge \mathbf{X}^{-1} )^\vee \f$
   */
  TangentMap Ad() const noexcept
  {
    TangentMap ret;
    Impl::Ad(cderived().coeffs(), ret);
    return ret;
  }

  /**
   * @brief Right-plus.
   *
   * @return \f$ \mathbf{x} \oplus \mathbf{a} = \mathbf{x} \circ \exp(\mathbf{a}) \f$
   */
  template<typename TangentDerived>
  PlainObject operator+(const Eigen::MatrixBase<TangentDerived> & a) const noexcept
  {
    return *this * exp(a);
  }

  /**
   * @brief Inplace right-plus: \f$ \mathbf{x} \leftarrow \mathbf{x} \circ \exp(\mathbf{a}) \f$.
   *
   * @return Reference to this
   */
  template<typename TangentDerived>
    requires(is_mutable)
  Derived & operator+=(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    *this *= exp(a);
    return derived();
  }

  /**
   * @brief Right-minus.
   *
   * @return Tangent space element \f$ \mathbf{a} \f$ such that \f$ \mathbf{this} = \mathbf{xo}
   * \oplus \mathbf{a} \f$.
   *
   * \f[
   *   \mathbf{x}_1 \ominus \mathbf{x}_2 := \log(\mathbf{x}_2^{-1} \circ \mathbf{x}_1)
   * \f]
   */
  template<typename OtherDerived>
    requires(std::is_same_v<Impl, typename liebase_info<OtherDerived>::Impl>)
  Tangent operator-(const LieGroupBase<OtherDerived> & xo) const noexcept
  {
    return (xo.inverse() * *this).log();
  }

  // Tangent API

  /**
   * @brief Lie group exponential map.
   */
  template<typename TangentDerived>
  static PlainObject exp(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    PlainObject ret;
    Impl::exp(a, ret.coeffs());
    return ret;
  }

  /**
   * @brief Lie algebra hat map.
   *
   * Maps a parameterization \f$ a \in \mathbb{R}^\mathtt{dof} \f$ a matrix Lie algebra element
   * to the corresponding matrix Lie algebra element \f$ A \in \mathbb{R}^{\mathtt{dim} \times
   * \mathtt{dim}} \f$.
   *
   * @see vee for the inverse of hat
   */
  template<typename TangentDerived>
  static Matrix hat(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    Matrix ret;
    Impl::hat(a, ret);
    return ret;
  }

  /**
   * @brief Lie alebra vee map.
   *
   * Maps a matrix Lie algebra element \f$ A \in \mathbb{R}^{\texttt{dim} \times \mathtt{dim}} \f$
   * to the corresponding parameterization \f$ a \in \mathbb{R}^\mathtt{dof} \f$.
   *
   * @see hat for the inverse of vee
   */
  template<typename MatrixDerived>
  static Tangent vee(const Eigen::MatrixBase<MatrixDerived> & A) noexcept
  {
    Tangent ret;
    Impl::vee(A, ret);
    return ret;
  }

  /**
   * @brief Lie algebra adjoint.
   *
   * @return Matrix \f$ \mathbf{ad}_\mathbf{a} \f$ s.t. \f$ \mathbf{ad}_\mathbf{a} \mathbf{b} =
   * [\mathbf{a}, \mathbf{b}] \f$
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    TangentMap ret;
    Impl::ad(a, ret);
    return ret;
  }

  /**
   * @brief Lie algebra bracket.
   *
   * \f[
   * [ \mathbf{a}, \mathbf{b}] = \left( \mathbf{a}^\wedge \mathbf{b}^\wedge - \mathbf{b}^\wedge
   * \mathbf{a}^\wedge \right)^\vee. \f]
   */
  template<typename TangentDerived1, typename TangentDerived2>
  static Tangent lie_bracket(
    const Eigen::MatrixBase<TangentDerived1> & a,
    const Eigen::MatrixBase<TangentDerived2> & b) noexcept
  {
    return ad(a) * b;
  }
  /**
   * @brief Right jacobian of the exponential map.
   *
   * @return \f$ \mathrm{d}^r \exp_a \f$
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    TangentMap ret;
    Impl::dr_exp(a, ret);
    return ret;
  }

  /**
   * @brief Inverse of right jacobian of the exponential map.
   *
   * @return \f$ \left( \mathrm{d}^r \exp_a \right)^{-1} \f$
   */
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    TangentMap ret;
    Impl::dr_expinv(a, ret);
    return ret;
  }

  /**
   * @brief Left jacobian of the exponential map.
   *
   * @return \f$ \mathrm{d}^l \exp_a \f$
   */
  template<typename TangentDerived>
  static TangentMap dl_exp(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    return dr_exp(-a);
  }

  /**
   * @brief Inverse of left jacobian of the exponential map.
   *
   * @return \f$ \left( \mathrm{d}^l \exp_a \right)^{-1} \f$
   */
  template<typename TangentDerived>
  static TangentMap dl_expinv(const Eigen::MatrixBase<TangentDerived> & a) noexcept
  {
    return dr_expinv(-a);
  }
};

/**
 * @brief Stream operator for Lie groups that ouputs the coefficients.
 */
template<typename Stream, typename Derived>
Stream & operator<<(Stream & s, const LieGroupBase<Derived> & g)
{
  s << static_cast<const Derived &>(g).coeffs().transpose();
  return s;
}

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__LIE_GROUP_BASE_HPP_
