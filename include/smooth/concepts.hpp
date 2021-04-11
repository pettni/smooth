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
  {s.template cast<float>()};
  {s.template cast<double>()};
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
  typename T::Vector;
  typename T::TangentMap;
  typename T::Algebra;
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
(T::Tangent::RowsAtCompileTime == T::dof) &&
(T::Tangent::ColsAtCompileTime == 1) &&
(T::Vector::RowsAtCompileTime == T::dim) &&
(T::Vector::ColsAtCompileTime == 1) &&
(T::TangentMap::RowsAtCompileTime == T::dof) &&
(T::TangentMap::ColsAtCompileTime == T::dof) &&
(T::Algebra::RowsAtCompileTime == T::dim) &&
(T::Algebra::ColsAtCompileTime == T::dim) &&
(!StorageLike<typename T::Storage, typename T::Scalar,
T::size>|| requires(T & t, std::default_random_engine & rng)
  {
    // group non-const interface
    {t.setIdentity()}->std::same_as<void>;
    {t.setRandom(rng)}->std::same_as<void>;
    {t.coeffs()}->std::same_as<typename T::Storage &>;
    {t.data()}->std::same_as<typename T::Scalar *>;
  }) &&
requires(const T & t, const T & u, const typename T::Vector & x)
{
  // group const interface
  {t.coeffs()}->std::same_as<const typename T::Storage &>;
  {t.data()}->std::same_as<const typename T::Scalar *>;
  {t.inverse()}->std::same_as<typename T::Group>;
  {t * x}->std::same_as<typename T::Vector>;
  {t * u}->std::same_as<typename T::Group>;
  {t.log()}->std::same_as<typename T::Tangent>;
  {t.Ad()}->std::same_as<typename T::TangentMap>;
  {t.template cast<double>()}->std::same_as<change_template_args_t<typename T::Group, double,
    DefaultStorage<double, T::size>>>;
  {t.template cast<float>()}->std::same_as<change_template_args_t<typename T::Group, float,
    DefaultStorage<float, T::size>>>;
} &&
requires(const typename T::Tangent & t) {
  // tangent const interface
  {T::exp(t)}->std::same_as<change_template_args_t<
      T, typename T::Scalar, DefaultStorage<typename T::Scalar, T::size>
    >>;
  {T::ad(t)}->std::same_as<typename T::TangentMap>;
  {T::hat(t)}->std::same_as<typename T::Algebra>;
};

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_
