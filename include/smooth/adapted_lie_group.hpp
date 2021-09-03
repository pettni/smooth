#ifndef SMOOTH__ADAPTED_LIE_GROUP_HPP_
#define SMOOTH__ADAPTED_LIE_GROUP_HPP_

#include <Eigen/Core>
#include <concepts>

#include "concepts.hpp"

/**
 * @file adapted_lie_group.hpp Lie group interface for third-party types.
 */

namespace smooth {

/**
 * @brief Trait class for making a class an AdaptedLieGroup instance
 */
template<typename T>
struct lie;

// clang-format off

/**
 * @brief Class-external Lie group interface defined via the lie trait.
 */
template<typename G>
concept AdaptedLieGroup =
requires {
  typename lie<G>::Scalar;
  typename lie<G>::PlainObject;
} &&
requires {
  {lie<G>::Dim}->std::convertible_to<Eigen::Index>;
  {lie<G>::Dof}->std::convertible_to<Eigen::Index>;
  {lie<G>::Identity()}->std::convertible_to<typename lie<G>::PlainObject>;
  {lie<G>::Random()}->std::convertible_to<typename lie<G>::PlainObject>;
} &&
requires(const G & g1, const G & g2, typename lie<G>::Scalar eps) {
  {lie<G>::Ad(g1)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, lie<G>::Dof>>;
  {lie<G>::composition(g1, g2)}->std::convertible_to<typename lie<G>::PlainObject>;
  {lie<G>::dim(g1)}->std::convertible_to<Eigen::Index>;
  {lie<G>::dof(g1)}->std::convertible_to<Eigen::Index>;
  {lie<G>::inverse(g1)}->std::convertible_to<typename lie<G>::PlainObject>;
  {lie<G>::isApprox(g1, g2, eps)}->std::convertible_to<bool>;
  {lie<G>::log(g1)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, 1>>;
  {lie<G>::matrix(g1)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dim, lie<G>::Dim>>;
  {lie<G>::template cast<double>(g1)};
  {lie<G>::template cast<float>(g1)};
} &&
requires(const Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, 1> & a) {
  {lie<G>::ad(a)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, lie<G>::Dof>>;
  {lie<G>::exp(a)}->std::convertible_to<typename lie<G>::PlainObject>;
  {lie<G>::hat(a)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dim, lie<G>::Dim>>;
  {lie<G>::dr_exp(a)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, lie<G>::Dof>>;
  {lie<G>::dr_expinv(a)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, lie<G>::Dof>>;
} &&
requires(const Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dim, lie<G>::Dim> & A) {
  {lie<G>::vee(A)}->std::convertible_to<Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, 1>>;
};

// clang-format on

////////////////////////////////////////////////
//// Free functions that dispatch to traits ////
////////////////////////////////////////////////

// Static constants

template<AdaptedLieGroup G>
static constexpr Eigen::Index Dof = lie<G>::Dof;

template<AdaptedLieGroup G>
static constexpr Eigen::Index Dim = lie<G>::Dim;

// Types

/**
 * @brief Group type
 */
template<AdaptedLieGroup G>
using Scalar = typename lie<G>::Scalar;

/**
 * @brief Group type
 */
template<AdaptedLieGroup G>
using PlainObject = typename lie<G>::PlainObject;

/**
 * @brief Vector of size Dof
 */
template<AdaptedLieGroup G>
using Tangent = Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, 1>;

/**
 * @brief Matrix of size Dim x Dim
 */
template<AdaptedLieGroup G>
using Matrix = Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dim, lie<G>::Dim>;

/**
 * @brief Matrix of size Dof x Dof
 */
template<AdaptedLieGroup G>
using TangentMap = Eigen::Matrix<typename lie<G>::Scalar, lie<G>::Dof, lie<G>::Dof>;

// Group interface

/**
 * @brief Group identity element
 */
template<AdaptedLieGroup G>
PlainObject<G> Identity()
{
  return lie<G>::Identity();
}

/**
 * @brief Random group element
 */
template<AdaptedLieGroup G>
PlainObject<G> Random()
{
  return lie<G>::Random();
}

/**
 * @brief Group adjoint \f$ Ad_g a \coloneq (G * \hat(a) * G^{-1})^{\wedge} \f$
 */
template<AdaptedLieGroup G>
TangentMap<G> Ad(const G & g)
{
  return lie<G>::Ad(g);
}

/**
 * @brief Group binary composition
 */
template<AdaptedLieGroup G1, AdaptedLieGroup G2>
PlainObject<G1> composition(const G1 & g, const G2 & g2)
{
  return lie<G1>::composition(g, g2);
}

/**
 * @brief Group multinary composition
 */
template<AdaptedLieGroup G1, AdaptedLieGroup G2, AdaptedLieGroup... Gs>
PlainObject<G1> composition(const G1 & g1, const G2 & g2, const Gs &... gs)
{
  return composition(composition(g1, g2), gs...);
}

/**
 * @brief Degrees of freedom of Lie group
 */
template<AdaptedLieGroup G>
Eigen::Index dof(const G & g)
{
  return lie<G>::dof(g);
}

/**
 * @brief Matrix dimension of Lie group
 */
template<AdaptedLieGroup G>
Eigen::Index dim(const G & g)
{
  return lie<G>::dim(g);
}

/**
 * @brief Group inverse
 */
template<AdaptedLieGroup G>
auto inverse(const G & g)
{
  return lie<G>::inverse(g);
}

/**
 * @brief Check if two group elements are approximately equal
 */
template<AdaptedLieGroup G1, AdaptedLieGroup G2>
bool isApprox(
  const G1 & g1, const G2 & g2, Scalar<G1> eps = Eigen::NumTraits<Scalar<G1>>::dummy_precision())
{
  return lie<G1>::isApprox(g1, g2, eps);
}

/**
 * @brief Group logarithm
 *
 * @see exp
 */
template<AdaptedLieGroup G>
Tangent<G> log(const G & g)
{
  return lie<G>::log(g);
}

/**
 * @brief Matrix Lie group representation
 */
template<AdaptedLieGroup G>
Matrix<G> matrix(const G & g)
{
  return lie<G>::matrix(g);
}

/**
 * @brief Cast to different scalar type
 */
template<typename NewScalar, AdaptedLieGroup G>
auto cast(const G & g)
{
  return lie<G>::template cast<NewScalar>(g);
}

// Tangent interface

/**
 * @brief Lie algebra adjoint \f$ ad_a b = [a, b] \f$
 */
template<AdaptedLieGroup G>
TangentMap<G> ad(const Tangent<G> & a)
{
  return lie<G>::ad(a);
}

/**
 * @brief Lie algebra exponential
 *
 * @see log
 */
template<AdaptedLieGroup G, typename Derived>
G exp(const Eigen::MatrixBase<Derived> & a)
{
  return lie<G>::exp(a);
}

/**
 * @brief Lie algebra hat map (maps from tangent space to matrix Lie algebra)
 *
 * @see vee
 */
template<AdaptedLieGroup G>
Matrix<G> hat(const Tangent<G> & a)
{
  return lie<G>::hat(a);
}

/**
 * @brief Lie algebra vee map (maps from matrix Lie algebra to tangent space)
 *
 * @see hat
 */
template<AdaptedLieGroup G>
Tangent<G> vee(const Matrix<G> & A)
{
  return lie<G>::vee(A);
}

/**
 * @brief Right derivative of exponential map
 */
template<AdaptedLieGroup G>
TangentMap<G> dr_exp(const Tangent<G> & a)
{
  return lie<G>::dr_exp(a);
}

/**
 * @brief Right derivative of exponential map inverse
 */
template<AdaptedLieGroup G>
TangentMap<G> dr_expinv(const Tangent<G> & a)
{
  return lie<G>::dr_expinv(a);
}

// Convenience methods

/**
 * @brief Right-plus
 */
template<AdaptedLieGroup G, typename Derived>
auto rplus(const G & g, const Eigen::MatrixBase<Derived> & a)
{
  return composition(g, ::smooth::exp<G>(a));
}

/**
 * @brief Left-plus
 */
template<AdaptedLieGroup G, typename Derived>
auto lplus(const G & g, const Eigen::MatrixBase<Derived> & a)
{
  return composition(::smooth::exp<G>(a), g);
}

/**
 * @brief Right-minus
 */
template<AdaptedLieGroup G>
auto rsub(const G & g1, const G & g2)
{
  return log(composition(inverse(g2), g1));
}

/**
 * @brief Left-minus
 */
template<AdaptedLieGroup G>
auto lsub(const G & g1, const G & g2)
{
  return log(composition(g1, inverse(g2)));
}

/**
 * @brief Left derivative of exponential map
 */
template<AdaptedLieGroup G, typename Derived>
TangentMap<G> dl_exp(const Eigen::MatrixBase<Derived> & a)
{
  return Ad(::smooth::exp<G>(a)) * dr_exp<G>(a);
}

/**
 * @brief Left derivative of exponential map inverse
 */
template<AdaptedLieGroup G, typename Derived>
TangentMap<G> dl_expinv(const Eigen::MatrixBase<Derived> & a)
{
  return -ad<G>(a) + dr_expinv<G>(a);
}

//////////////////////////////
//// Adapter for LieGroup ////
//////////////////////////////

/**
 * @brief External Lie group interface for native LieGroup's
 */
template<LieGroup G>
struct lie<G>
{
  // \cond
  using Scalar      = typename G::Scalar;
  using PlainObject = typename G::PlainObject;

  static constexpr Eigen::Index Dof = G::Dof;
  static constexpr Eigen::Index Dim = G::Dim;

  // group interface

  static PlainObject Identity() { return G::Identity(); }
  static PlainObject Random() { return G::Random(); }
  static typename G::TangentMap Ad(const G & g) { return g.Ad(); }
  template<LieGroup Go>
  static PlainObject composition(const G & g1, const Go & g2)
  {
    return g1.operator*(g2);
  }
  static Eigen::Index dof(const G &) { return G::Dof; }
  static Eigen::Index dim(const G &) { return G::Dim; }
  static PlainObject inverse(const G & g) { return g.inverse(); }
  template<LieGroup Go>
  static bool isApprox(const G & g, const Go & go, Scalar eps)
  {
    return g.isApprox(go, eps);
  }
  static typename G::Tangent log(const G & g) { return g.log(); }
  static typename G::Matrix matrix(const G & g) { return g.matrix(); }
  template<typename NewScalar>
  static auto cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static typename G::TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  {
    return G::ad(a);
  }
  template<typename Derived>
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::exp(a);
  }
  template<typename Derived>
  static typename G::Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    return G::hat(a);
  }
  template<typename Derived>
  static typename G::Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return G::vee(A);
  }
  template<typename Derived>
  static typename G::TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_exp(a);
  }
  template<typename Derived>
  static typename G::TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_expinv(a);
  }
  // \endcond
};

///////////////////////////////////
//// Wrapper for Eigen vectors ////
///////////////////////////////////

/**
 * @brief Concept to identify Eigen column vectors
 */
template<typename G>
concept RnType = std::is_base_of_v<Eigen::MatrixBase<G>, G> && G::ColsAtCompileTime == 1;

/**
 * @brief External Lie group interface for Eigen vectors
 */
template<RnType G>
struct lie<G>
{
  // \cond
  using Scalar      = typename G::Scalar;
  using PlainObject = typename G::PlainObject;

  static constexpr int Dof = G::SizeAtCompileTime;
  static constexpr int Dim = Dof == -1 ? -1 : Dof + 1;

private:
  using Matrix     = Eigen::Matrix<Scalar, Dim, Dim>;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;

public:
  // group interface

  static PlainObject Identity() { return G::Zero(); }
  static PlainObject Random() { return G::Random(); }
  static TangentMap Ad(const G &) { return TangentMap::Identity(); }
  template<typename Derived>
  static PlainObject composition(const G & g1, const Eigen::MatrixBase<Derived> & g2)
  {
    return g1 + g2;
  }
  static Eigen::Index dof(const G & g) { return g.size(); }
  static Eigen::Index dim(const G & g) { return g.size(); }
  static PlainObject inverse(const G & g) { return -g; }
  template<typename Derived>
  static bool isApprox(const G & g, const Eigen::MatrixBase<Derived> & g2, Scalar eps)
  {
    return g.isApprox(g2, eps);
  }
  static Tangent log(const G & g) { return g; }
  static Matrix matrix(const G & g)
  {
    Matrix ret                            = Matrix::Identity();
    ret.template topRightCorner<Dof, 1>() = g;
    return ret;
  }
  template<typename NewScalar>
  static auto cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Zero();
  }
  template<typename Derived>
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a;
  }
  template<typename Derived>
  static Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    Matrix ret                            = Matrix::Zero();
    ret.template topRightCorner<Dof, 1>() = a;
    return ret;
  }
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return A.template topRightCorner<Dof, 1>();
  }
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  // \endcond
};

///////////////////////////////////////////////////
//// Wrapper for built-in floating point types ////
///////////////////////////////////////////////////

/**
 * @brief Concept to identify built-in scalars
 */
template<typename G>
concept FloatingType = std::is_floating_point_v<G>;

/**
 * @brief External Lie group interface for built-in floating point types
 */
template<FloatingType G>
struct lie<G>
{
  // \cond
  using Scalar      = G;
  using PlainObject = G;

  static constexpr int Dof = 1;
  static constexpr int Dim = 2;

private:
  using Matrix     = Eigen::Matrix<Scalar, Dim, Dim>;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;

public:
  // group interface

  static PlainObject Identity() { return G(0); }
  static PlainObject Random()
  {
    return G(Scalar(-1) + static_cast<Scalar>(rand()) / static_cast<Scalar>(RAND_MAX / 2));
  }
  static TangentMap Ad(G) { return TangentMap{1}; }
  static PlainObject composition(G g1, G g2) { return g1 + g2; }
  static Eigen::Index dof(G) { return 1; }
  static Eigen::Index dim(G) { return 2; }
  static PlainObject inverse(G g) { return -g; }
  static bool isApprox(G g1, G g2, Scalar eps)
  {
    using std::abs;
    return abs<G>(g1 - g2) <= eps * abs<G>(g1);
  }
  static Tangent log(G g) { return Tangent{g}; }
  static Matrix matrix(G g) { return Eigen::Matrix2<Scalar>{{{1, g}, {0, 1}}}; }
  template<typename NewScalar>
  static NewScalar cast(G g)
  {
    return static_cast<NewScalar>(g);
  }

  // tangent interface

  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Zero();
  }
  template<typename Derived>
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a(0);
  }
  template<typename Derived>
  static Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix2<Scalar>{
      {0, a(0)},
      {0, 0},
    };
  }
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return Tangent{A(0, 1)};
  }
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }

  // \endcond
};

}  // namespace smooth

#endif  // SMOOTH__ADAPTED_LIE_GROUP_HPP_
