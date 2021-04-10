#ifndef SMOOTH__CONCEPTS_HPP_
#define SMOOTH__CONCEPTS_HPP_

#include <Eigen/Core>

#include <concepts>

#include "common.hpp"
#include "traits.hpp"


namespace smooth
{

template<typename S, typename Scalar, std::size_t N>
concept ConstStorageLike = requires {
  typename S::Scalar;
  {S::SizeAtCompileTime}->std::convertible_to<uint32_t>;
} &&
std::is_arithmetic_v<typename S::Scalar>&&
std::is_same_v<Scalar, typename S::Scalar>&&
N == S::SizeAtCompileTime &&
requires(const S & s, int i) {
  {s[i]}->std::convertible_to<typename S::Scalar>;
  {s.data()}->std::same_as<const typename S::Scalar *>;
};


template<typename S, typename Scalar, std::size_t N>
concept StorageLike = ConstStorageLike<S, Scalar, N>&&
  requires(S & s, int i) {
  {s[i]}->std::same_as<typename S::Scalar &>;
  {s.data()}->std::same_as<typename S::Scalar *>;
};


template<typename T>
concept LieGroupLike = requires {
  typename T::Storage;
  typename T::Scalar;
  typename T::Group;
  typename T::Tangent;
  typename T::TangentMap;
  typename T::Algebra;
  {T::size}->std::same_as<const uint32_t &>;
  {T::dim}->std::same_as<const uint32_t &>;
  {T::dof}->std::same_as<const uint32_t &>;
} &&
std::is_base_of_v<Eigen::MatrixBase<typename T::Tangent>, typename T::Tangent>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::TangentMap>, typename T::TangentMap>&&
std::is_base_of_v<Eigen::MatrixBase<typename T::Algebra>, typename T::Algebra>&&
std::is_same_v<typename T::Tangent::Scalar, typename T::Scalar>&&
std::is_same_v<
  typename T::Group,
  change_template_args_t<T, typename T::Scalar, DefaultStorage<typename T::Scalar, T::size>>
>&&
(T::Tangent::RowsAtCompileTime == T::dof) &&
(T::Tangent::ColsAtCompileTime == 1) &&
(T::TangentMap::RowsAtCompileTime == T::dof) &&
(T::TangentMap::ColsAtCompileTime == T::dof) &&
(T::Algebra::RowsAtCompileTime == T::dim) &&
(T::Algebra::ColsAtCompileTime == T::dim) &&
requires(T & t)
{
  {t.setIdentity()}->std::same_as<void>;
} &&
requires(const T & t)
{
  {t.log()}->std::same_as<typename T::Tangent>;
  {t.Ad()}->std::same_as<typename T::TangentMap>;
} &&
requires(const typename T::Tangent & t) {
  {T::exp(t)}->std::same_as<change_template_args_t<T, typename T::Scalar,
    DefaultStorage<typename T::Scalar, T::size>>>;
  {T::ad(t)}->std::same_as<typename T::TangentMap>;
  {T::hat(t)}->std::same_as<typename T::Algebra>;
};

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_
