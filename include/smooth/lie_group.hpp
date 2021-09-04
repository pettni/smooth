#ifndef SMOOTH__ADAPTED_LIE_GROUP_HPP_
#define SMOOTH__ADAPTED_LIE_GROUP_HPP_

#include <Eigen/Core>
#include <concepts>

#include "manifold.hpp"

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
 * @brief Class with an internally defined Lie group interface.
 */
template<typename G>
concept NativeLieGroup =
// static constants
requires {
  typename G::Scalar;
  typename G::Tangent;
  typename G::PlainObject;
  {G::Dof}->std::convertible_to<Eigen::Index>;
  {G::Dim}->std::convertible_to<Eigen::Index>;
  {G::Identity()}->std::convertible_to<typename G::PlainObject>;
  {G::Random()}->std::convertible_to<typename G::PlainObject>;
} &&
(G::Dof >= 1) &&
(G::Tangent::SizeAtCompileTime == G::Dof) &&
// member methods
requires(const G & g1, const G & g2, typename G::Scalar eps)
{
  {g1.dof()}->std::convertible_to<Eigen::Index>;             // degrees of freedom at runtime
  {g1.Ad()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {g1 * g2}->std::convertible_to<typename G::PlainObject>;
  {g1.inverse()}->std::convertible_to<typename G::PlainObject>;
  {g1.isApprox(g2, eps)}->std::convertible_to<bool>;
  {g1.log()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, 1>>;
  {g1.matrix()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dim, G::Dim>>;
  {g1.template cast<float>()};
  {g1.template cast<double>()};
} &&
// static methods
requires(const Eigen::Matrix<typename G::Scalar, G::Dof, 1> & a)
{
  {G::ad(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::exp(a)}->std::convertible_to<typename G::PlainObject>;
  {G::hat(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dim, G::Dim>>;
  {G::dr_exp(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::dr_expinv(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
} &&
requires(const Eigen::Matrix<typename G::Scalar, G::Dim, G::Dim> & A)
{
  {G::vee(A)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, 1>>;
};

// clang-format on

/**
 * @brief External Lie group interface for native LieGroup's
 */
template<NativeLieGroup G>
struct lie<G>
{
  // \cond
  using Scalar      = typename G::Scalar;
  using PlainObject = typename G::PlainObject;

  static constexpr Eigen::Index Dof = G::Dof;
  static constexpr Eigen::Index Dim = G::Dim;

  // group interface

  static inline PlainObject Identity() { return G::Identity(); }
  static inline PlainObject Random() { return G::Random(); }
  static inline typename G::TangentMap Ad(const G & g) { return g.Ad(); }
  template<NativeLieGroup Go>
  static inline PlainObject composition(const G & g1, const Go & g2)
  {
    return g1.operator*(g2);
  }
  static inline Eigen::Index dof(const G &) { return G::Dof; }
  static inline Eigen::Index dim(const G &) { return G::Dim; }
  static inline PlainObject inverse(const G & g) { return g.inverse(); }
  template<NativeLieGroup Go>
  static inline bool isApprox(const G & g, const Go & go, Scalar eps)
  {
    return g.isApprox(go, eps);
  }
  static inline typename G::Tangent log(const G & g) { return g.log(); }
  static inline typename G::Matrix matrix(const G & g) { return g.matrix(); }
  template<typename NewScalar>
  static inline auto cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static inline typename G::TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  {
    return G::ad(a);
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::exp(a);
  }
  template<typename Derived>
  static inline typename G::Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    return G::hat(a);
  }
  template<typename Derived>
  static inline typename G::Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return G::vee(A);
  }
  template<typename Derived>
  static inline typename G::TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_exp(a);
  }
  template<typename Derived>
  static inline typename G::TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    return G::dr_expinv(a);
  }
  // \endcond
};

// clang-format off

/**
 * @brief Class-external Lie group interface defined via the lie trait.
 */
template<typename G>
concept AdaptedLieGroup =
std::is_default_constructible_v<G> &&
std::is_copy_constructible_v<G> &&
std::is_copy_assignable_v<G> &&
requires {
  typename lie<G>::Scalar;
  typename lie<G>::PlainObject;
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
static inline constexpr Eigen::Index Dim = lie<G>::Dim;

// Types

/**
 * @brief Group type
 */
template<AdaptedLieGroup G>
using PlainObject = typename lie<G>::PlainObject;

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
inline auto Identity()
{
  return lie<G>::Identity();
}

/**
 * @brief Random group element
 */
template<AdaptedLieGroup G>
inline auto Random()
{
  return lie<G>::Random();
}

/**
 * @brief Group adjoint \f$ Ad_g a \coloneq (G * \hat(a) * G^{-1})^{\wedge} \f$
 */
template<AdaptedLieGroup G>
inline auto Ad(const G & g)
{
  return lie<G>::Ad(g);
}

/**
 * @brief Group binary composition
 */
template<AdaptedLieGroup G, typename Arg>
inline auto composition(const G & g, Arg && a)
{
  return lie<G>::composition(g, std::forward<Arg>(a));
}

/**
 * @brief Group multinary composition
 */
template<AdaptedLieGroup G, typename Arg, typename... Args>
inline auto composition(const G & g, Arg && a, Args &&... as)
{
  return composition(composition(g, std::forward<Arg>(a)), std::forward<Args>(as)...);
}

/**
 * @brief Matrix dimension of Lie group
 */
template<AdaptedLieGroup G>
inline auto dim(const G & g)
{
  return lie<G>::dim(g);
}

/**
 * @brief Group inverse
 */
template<AdaptedLieGroup G>
inline auto inverse(const G & g)
{
  return lie<G>::inverse(g);
}

/**
 * @brief Check if two group elements are approximately equal
 */
template<AdaptedLieGroup G, typename Arg>
inline auto isApprox(
  const G & g, Arg && a, Scalar<G> eps = Eigen::NumTraits<Scalar<G>>::dummy_precision())
{
  return lie<G>::isApprox(g, std::forward<Arg>(a), eps);
}

/**
 * @brief Group logarithm
 *
 * @see exp
 */
template<AdaptedLieGroup G>
inline auto log(const G & g)
{
  return lie<G>::log(g);
}

/**
 * @brief Matrix Lie group representation
 */
template<AdaptedLieGroup G>
inline auto matrix(const G & g)
{
  return lie<G>::matrix(g);
}

// Tangent interface

/**
 * @brief Lie algebra adjoint \f$ ad_a b = [a, b] \f$
 */
template<AdaptedLieGroup G, typename Arg>
inline auto ad(Arg && a)
{
  return lie<G>::ad(std::forward<Arg>(a));
}

/**
 * @brief Lie algebra exponential
 *
 * @see log
 */
template<AdaptedLieGroup G, typename Arg>
inline auto exp(Arg && a)
{
  return lie<G>::exp(std::forward<Arg>(a));
}

/**
 * @brief Lie algebra hat map (maps from tangent space to matrix Lie algebra)
 *
 * @see vee
 */
template<AdaptedLieGroup G, typename Arg>
inline auto hat(Arg && a)
{
  return lie<G>::hat(std::forward<Arg>(a));
}

/**
 * @brief Lie algebra vee map (maps from matrix Lie algebra to tangent space)
 *
 * @see hat
 */
template<AdaptedLieGroup G, typename Arg>
inline auto vee(Arg && A)
{
  return lie<G>::vee(std::forward<Arg>(A));
}

/**
 * @brief Right derivative of exponential map
 */
template<AdaptedLieGroup G, typename Arg>
inline auto dr_exp(Arg && a)
{
  return lie<G>::dr_exp(std::forward<Arg>(a));
}

/**
 * @brief Right derivative of exponential map inverse
 */
template<AdaptedLieGroup G, typename Arg>
inline auto dr_expinv(Arg && a)
{
  return lie<G>::dr_expinv(std::forward<Arg>(a));
}

// Convenience methods

/**
 * @brief Left-plus
 */
template<AdaptedLieGroup G, typename Derived>
inline PlainObject<G> lplus(const G & g, const Eigen::MatrixBase<Derived> & a)
{
  return composition(::smooth::exp<G>(a), g);
}

/**
 * @brief Left-minus
 */
template<AdaptedLieGroup G>
inline Tangent<G> lsub(const G & g1, const G & g2)
{
  return log(composition(g1, inverse(g2)));
}

/**
 * @brief Left derivative of exponential map
 */
template<AdaptedLieGroup G, typename Derived>
inline TangentMap<G> dl_exp(const Eigen::MatrixBase<Derived> & a)
{
  return Ad(::smooth::exp<G>(a)) * dr_exp<G>(a);
}

/**
 * @brief Left derivative of exponential map inverse
 */
template<AdaptedLieGroup G, typename Derived>
inline TangentMap<G> dl_expinv(const Eigen::MatrixBase<Derived> & a)
{
  return -ad<G>(a) + dr_expinv<G>(a);
}

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

  static inline PlainObject Identity() { return G::Zero(); }
  static inline PlainObject Random() { return G::Random(); }
  static inline TangentMap Ad(const G &) { return TangentMap::Identity(); }
  template<typename Derived>
  static inline PlainObject composition(const G & g1, const Eigen::MatrixBase<Derived> & g2)
  {
    return g1 + g2;
  }
  static inline Eigen::Index dof(const G & g) { return g.size(); }
  static inline Eigen::Index dim(const G & g) { return g.size(); }
  static inline PlainObject inverse(const G & g) { return -g; }
  template<typename Derived>
  static inline bool isApprox(const G & g, const Eigen::MatrixBase<Derived> & g2, Scalar eps)
  {
    return g.isApprox(g2, eps);
  }
  static inline Tangent log(const G & g) { return g; }
  static inline Matrix matrix(const G & g)
  {
    Matrix ret                            = Matrix::Identity();
    ret.template topRightCorner<Dof, 1>() = g;
    return ret;
  }
  template<typename NewScalar>
  static inline Eigen::Matrix<NewScalar, Dof, 1> cast(const G & g)
  {
    return g.template cast<NewScalar>();
  }

  // tangent interface

  template<typename Derived>
  static inline TangentMap ad(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Zero();
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a;
  }
  template<typename Derived>
  static inline Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    Matrix ret                            = Matrix::Zero();
    ret.template topRightCorner<Dof, 1>() = a;
    return ret;
  }
  template<typename Derived>
  static inline Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return A.template topRightCorner<Dof, 1>();
  }
  template<typename Derived>
  static inline TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static inline TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
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

  static inline PlainObject Identity() { return G(0); }
  static inline PlainObject Random()
  {
    return G(Scalar(-1) + static_cast<Scalar>(rand()) / static_cast<Scalar>(RAND_MAX / 2));
  }
  static inline TangentMap Ad(G) { return TangentMap{1}; }
  static inline PlainObject composition(G g1, G g2) { return g1 + g2; }
  static inline Eigen::Index dof(G) { return 1; }
  static inline Eigen::Index dim(G) { return 2; }
  static inline PlainObject inverse(G g) { return -g; }
  static inline bool isApprox(G g1, G g2, Scalar eps)
  {
    using std::abs;
    return abs<G>(g1 - g2) <= eps * abs<G>(g1);
  }
  static inline Tangent log(G g) { return Tangent{g}; }
  static inline Matrix matrix(G g) { return Eigen::Matrix2<Scalar>{{{1, g}, {0, 1}}}; }
  template<typename NewScalar>
  static inline NewScalar cast(G g)
  {
    return static_cast<NewScalar>(g);
  }

  // tangent interface

  template<typename Derived>
  static inline TangentMap ad(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Zero();
  }
  template<typename Derived>
  static inline PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return a(0);
  }
  template<typename Derived>
  static inline Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix2<Scalar>{
      {0, a(0)},
      {0, 0},
    };
  }
  template<typename Derived>
  static inline Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return Tangent{A(0, 1)};
  }
  template<typename Derived>
  static inline TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static inline TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }

  // \endcond
};

/**
 * @brief AdaptedManifold interface for AdaptedLieGroup
 */
template<AdaptedLieGroup G>
struct man<G>
{
  // \cond
  using Scalar                      = typename lie<G>::Scalar;
  static constexpr Eigen::Index Dof = lie<G>::Dof;

  static inline Eigen::Index dof(const G & g) { return lie<G>::dof(g); }

  template<typename NewScalar>
  static inline auto cast(const G & g)
  {
    return lie<G>::template cast<NewScalar>(g);
  }

  template<typename Derived>
  static inline G rplus(const G & g, const Eigen::MatrixBase<Derived> & a)
  {
    return lie<G>::composition(g, lie<G>::exp(a));
  }

  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const G & g1, const G & g2)
  {
    return lie<G>::log(lie<G>::composition(lie<G>::inverse(g2), g1));
  }
  // \endcond
};

}  // namespace smooth

#endif  // SMOOTH__ADAPTED_LIE_GROUP_HPP_
