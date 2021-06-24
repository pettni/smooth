#ifndef SMOOTH__CONCEPTS_HPP_
#define SMOOTH__CONCEPTS_HPP_

#include <concepts>

#include <Eigen/Core>


namespace smooth
{

/**
 * @brief A (smooth) manifold M requires the following
 *
 * - M::Scalar scalar type
 * - M::SizeAtCompileTime tangent space dimension (compile time, -1 if dynamic)
 * - M.size() : tangent space dimension (runtime)
 * - M + T -> M : geodesic addition
 * - M - M -> T : inverse of geodesic addition (in practice only used for infinitesimal values)
 *
 * Where T = Eigen::Matrix<Scalar, SizeAtCompileTime, 1> is the tangent type
 */
template<typename M>
concept Manifold =
requires
{
  typename M::Scalar;
  typename M::PlainObject;
  {M::SizeAtCompileTime}->std::convertible_to<Eigen::Index>;  // degrees of freedom at compile time
} &&
requires(const M & m1, const M & m2, const Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1> & a)
{
  {m1.size()}->std::convertible_to<Eigen::Index>;             // degrees of freedom at runtime
  {m1 + a}->std::convertible_to<typename M::PlainObject>;
  {m1 - m2}->std::convertible_to<Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1>>;
  {m1.template cast<double>()};
};

template<typename T>
concept RnLike = Manifold<T> &&
std::is_base_of_v<Eigen::MatrixBase<T>, T> &&
T::IsVectorAtCompileTime == 1 &&
T::ColsAtCompileTime == 1;

template<typename T>
concept StaticRnLike = RnLike<T> && T::RowsAtCompileTime >= 1;

/**
 * @brief A Lie Group is a smooth manifold that is also a group.
 * This concept requires the exp and log maps, and the upper and
 * lowercase adjoints
 */
template<typename G>
concept LieGroup = Manifold<G> &&
// static constants
requires {
  {G::Dof}->std::convertible_to<Eigen::Index>;      // degrees of freedom
} &&
(G::Dof >= 1) &&
(G::SizeAtCompileTime == G::Dof) &&
// member methods
requires(const G & g1, const G & g2)
{
  {g1.inverse()}->std::convertible_to<typename G::PlainObject>;
  {g1 * g2}->std::convertible_to<typename G::PlainObject>;
  {g1.log()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, 1>>;
  {g1.Ad()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
} &&
// static methods
requires(const Eigen::Matrix<typename G::Scalar, G::Dof, 1> & a)
{
  {G::Identity()}->std::convertible_to<typename G::PlainObject>;
  {G::exp(a)}->std::convertible_to<typename G::PlainObject>;
  {G::ad(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
};

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_
