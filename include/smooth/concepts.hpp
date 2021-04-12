#ifndef SMOOTH__CONCEPTS_HPP_
#define SMOOTH__CONCEPTS_HPP_

#include <Eigen/Core>

#include <concepts>

#include "common.hpp"
#include "traits.hpp"


namespace smooth
{

/**
 * @brief Storage concept: requires operator[] to access one of N ints
 *
 * @tparam S storage type
 * @tparam Scalar scalar type
 * @tparam N number of scalars in storage
 */
template<typename S, typename Scalar, std::size_t N>
concept StorageLike = requires {
  typename S::Scalar;
  {S::SizeAtCompileTime}->std::convertible_to<uint32_t>;
} &&
std::is_same_v<Scalar, typename S::Scalar>&&
N == S::SizeAtCompileTime &&
requires(const S & s, int i) {
  {s[i]}->std::convertible_to<typename S::Scalar>;
};

/**
 * @brief Storage that can be modified (non-const)
 */
template<typename S, typename Scalar, std::size_t N>
concept ModifiableStorageLike = StorageLike<S, Scalar, N>&&
  requires(S & s, int i) {
  {s[i]}->std::convertible_to<typename S::Scalar &>;
};

/**
 * @brief Storage that guarantees that content is contiguous in memory
 * and is ordered in the same way as in DefaultStorage
 */
template<typename S, typename Scalar, std::size_t N>
concept OrderedStorageLike = StorageLike<S, Scalar, N>&&
  requires(const S & s) {
  {s.data()}->std::same_as<const typename S::Scalar *>;
};

/**
 * @brief Modifiable storage that also guarantees that content is contiguous in memory
 * and is ordered in the same way as in DefaultStorage
 */
template<typename S, typename Scalar, std::size_t N>
concept OrderedModifiableStorageLike =
  ModifiableStorageLike<S, Scalar, N>&& OrderedStorageLike<S, Scalar, N>&&
  requires(S & s) {
  {s.data()}->std::same_as<typename S::Scalar *>;
};

/**
 * @brief Storage that is not modifiable
 */
template<typename S, typename Scalar, std::size_t N>
concept ConstStorageLike = StorageLike<S, Scalar, N>&& !ModifiableStorageLike<S, Scalar, N>;

/**
 * @brief Storage that is not ordered
 */
template<typename S, typename Scalar, std::size_t N>
concept UnorderedStorageLike = StorageLike<S, Scalar, N>&& !OrderedStorageLike<S, Scalar, N>;


template<typename T>
concept LieGroupLike = requires {
  typename T::Storage;
  typename T::Scalar;
  typename T::Tangent;
  typename T::Vector;
  typename T::TangentMap;
  typename T::Algebra;
  typename T::Group;
  typename T::MatrixGroup;
  {T::size}->std::same_as<const uint32_t &>;
  {T::dim}->std::same_as<const uint32_t &>;
  {T::dof}->std::same_as<const uint32_t &>;
} &&
std::is_base_of_v<Eigen::MatrixBase<typename T::Tangent>, typename T::Tangent>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::Vector>, typename T::Vector>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::TangentMap>, typename T::TangentMap>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::Algebra>, typename T::Algebra>&&
std::is_same_v<typename T::Tangent::Scalar, typename T::Scalar>&&
std::is_same_v<
  typename T::Group,
  change_template_args_t<T, typename T::Scalar, DefaultStorage<typename T::Scalar, T::size>>
>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::Algebra>, typename T::MatrixGroup>&&
(T::Tangent::RowsAtCompileTime == T::dof) &&
(T::Tangent::ColsAtCompileTime == 1) &&
(T::Vector::RowsAtCompileTime == T::dim) &&
(T::Vector::ColsAtCompileTime == 1) &&
(T::TangentMap::RowsAtCompileTime == T::dof) &&
(T::TangentMap::ColsAtCompileTime == T::dof) &&
(T::Algebra::RowsAtCompileTime == T::dim) &&
(T::Algebra::ColsAtCompileTime == T::dim) &&
(T::MatrixGroup::RowsAtCompileTime == T::dim) &&
(T::MatrixGroup::ColsAtCompileTime == T::dim) &&
// Modifiable interface
(
  !ModifiableStorageLike<typename T::Storage, typename T::Scalar, T::size>||
  requires(T & t, std::default_random_engine & rng)
  {
    // group non-const interface
    {t.setIdentity()}->std::same_as<void>;
    {t.setRandom(rng)}->std::same_as<void>;
    {t.coeffs()}->std::same_as<typename T::Storage &>;
  }
) &&
// Ordered interface
(
  !OrderedStorageLike<typename T::Storage, typename T::Scalar, T::size>||
  requires(const T & t)
  {
    {t.data()}->std::same_as<const typename T::Scalar *>;
  }
) &&
// Ordered modifiable interface
(
  !OrderedModifiableStorageLike<typename T::Storage, typename T::Scalar, T::size>||
  requires(T & t)
  {
    {t.data()}->std::same_as<typename T::Scalar *>;
  }
) &&
// Const interface
requires(const T & t, const T & u, const typename T::Vector & x)
{
  // group const interface
  {t.coeffs()}->std::same_as<const typename T::Storage &>;
  {t.inverse()}->std::same_as<typename T::Group>;
  {t * x}->std::same_as<typename T::Vector>;
  {t * u}->std::same_as<typename T::Group>;
  {t.log()}->std::same_as<typename T::Tangent>;
  {t.matrix()}->std::same_as<typename T::MatrixGroup>;
  {t.Ad()}->std::same_as<typename T::TangentMap>;
  {t.template cast<double>()}->std::same_as<change_template_args_t<typename T::Group, double,
    DefaultStorage<double, T::size>>>;
  {t.template cast<float>()}->std::same_as<change_template_args_t<typename T::Group, float,
    DefaultStorage<float, T::size>>>;
} &&
// Tangent interface
requires(const typename T::Tangent & t) {
  {T::exp(t)}->std::same_as<change_template_args_t<
      T, typename T::Scalar, DefaultStorage<typename T::Scalar, T::size>
    >>;
  {T::ad(t)}->std::same_as<typename T::TangentMap>;
  {T::hat(t)}->std::same_as<typename T::Algebra>;
};

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_
