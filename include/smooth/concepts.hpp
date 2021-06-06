#ifndef SMOOTH__CONCEPTS_HPP_
#define SMOOTH__CONCEPTS_HPP_

#include <concepts>

#include <Eigen/Core>


namespace smooth
{

/**********************
 *  STORAGE CONCEPTS  *
 **********************/

/**
 * @brief Storage concept: requires operator[] to access one of N ints
 *
 * @tparam S storage type
 * @tparam Scalar scalar type
 * @tparam N number of scalars in storage
 */
template<typename S>
concept StorageLike = requires {
  typename S::Scalar;
  {S::Size}->std::convertible_to<Eigen::Index>;
} &&
requires(const S & s, int i) {
  {s[i]}->std::convertible_to<typename S::Scalar>;
};

// Storage that can be mapped
template<typename S>
concept MappableStorageLike = StorageLike<S>&&
  requires(const S & s) {
  {s.data()}->std::same_as<const typename S::Scalar *>;
};

// Storage that can be both mapped and modified
template<typename S>
concept ModifiableStorageLike =
  MappableStorageLike<S>&&
  requires(S & s, int i) {
  {s.data()}->std::same_as<typename S::Scalar *>;
  {s[i]}->std::convertible_to<typename S::Scalar &>;
};

/**
 * @brief Storage that is not modifiable
 */
template<typename S>
concept ConstStorageLike = StorageLike<S>&& !ModifiableStorageLike<S>;


/********************
 *  SPACE CONCEPTS  *
 ********************/

/**
 * @brief A (smooth) manifold M requires a tangent type T and must support
 * the following operations
 *
 * - M + T -> M : geodesic addition
 * - M += T : in-place geodesic addition
 * - M - M -> T : inverse of geodesic addition (in practice only used for infinitesimal values)
 */
template<typename M>
concept Manifold =
requires
{
  typename M::Scalar;
  typename M::PlainObject;
  {M::SizeAtCompileTime}->std::convertible_to<Eigen::Index>;  // degrees of freedom at compile time
} &&
requires(const M & m1, M & m2, const Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1> & a)
{
  {m1.size()}->std::convertible_to<Eigen::Index>;             // degrees of freedom at runtime
  {m1 + a}->std::convertible_to<typename M::PlainObject>;
  {m2 += a}->std::convertible_to<typename M::PlainObject>;
  {m1 - m2}->std::convertible_to<Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1>>;
  {m1.template cast<float>()};
};

template<typename T>
concept RnLike = Manifold<T> &&
std::is_base_of_v<Eigen::MatrixBase<T>, T> &&
T::IsVectorAtCompileTime == 1 &&
T::ColsAtCompileTime == 1;

template<typename T>
concept StaticRnLike = RnLike<T> && T::RowsAtCompileTime >= 1;

/**
 * @brief A (matrix) Lie Group is a smooth manifold that can be represented
 * as a subset of GL(F, c)
 */
template<typename G>
concept LieGroup = Manifold<G> &&
// static constants
requires {
  typename G::Tangent;
  {G::RepSize}->std::convertible_to<Eigen::Index>;  // representation size
  {G::Dof}->std::convertible_to<Eigen::Index>;      // degrees of freedom
  {G::Dim}->std::convertible_to<Eigen::Index>;      // dimension
  {G::ActDim}->std::convertible_to<Eigen::Index>;   // dimension of space of which group act on
} &&
(G::RepSize >= 1) &&
(G::Dof >= 1) &&
(G::Dim >= 1) &&
(G::ActDim >= 1) &&
(G::SizeAtCompileTime == G::Dof) &&
(std::is_same_v<typename G::Tangent, Eigen::Matrix<typename G::Scalar, G::Dof, 1>>) &&
// member methods
requires(const G & g1, const G & g2, const Eigen::Matrix<typename G::Scalar, G::ActDim, 1> & v)
{
  {g1.template cast<double>()};
  {g1.template cast<float>()};
  {g1.inverse()}->std::convertible_to<typename G::PlainObject>;
  {g1 * g2}->std::convertible_to<typename G::PlainObject>;
  {g1 * v}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::ActDim, 1>>;
  {g1.log()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, 1>>;
  {g1.matrix_group()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dim, G::Dim>>;
  {g1.Ad()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
} &&
// static methods
requires(const Eigen::Matrix<typename G::Scalar, G::Dof, 1> & a)
{
  {G::Identity()}->std::convertible_to<typename G::PlainObject>;
  {G::Random()}->std::convertible_to<typename G::PlainObject>;
  {G::exp(a)}->std::convertible_to<typename G::PlainObject>;
  {G::ad(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
  {G::hat(a)}->std::convertible_to<typename G::MatrixGroup>;
};

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_
