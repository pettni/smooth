#ifndef SMOOTH__STORAGE_HPP_
#define SMOOTH__STORAGE_HPP_

#include <Eigen/Core>

#include <cstdint>

#include "concepts.hpp"
#include "traits.hpp"


namespace smooth
{

/**
 * @brief Generic map type
 */
template<LieGroupLike G>
using Map = change_template_args_t<
  G,
  typename G::Scalar,
  Eigen::Map<DefaultStorage<typename G::Scalar, G::size>>
>;

/**
 * @brief Generic const map type
 */
template<LieGroupLike G>
using ConstMap = change_template_args_t<
  G,
  typename G::Scalar,
  const Eigen::Map<const DefaultStorage<typename G::Scalar, G::size>>
>;

/**
 * @brief Trait that specifies that a storage is ordered
 */
template<typename Storage>
struct is_ordered : public std::false_type
{};

// Default storage is ordered
template<typename Scalar, int Size>
struct is_ordered<DefaultStorage<Scalar, Size>> : public std::true_type
{};

// Map storage is ordered
template<typename Scalar, int Size>
struct is_ordered<Eigen::Map<DefaultStorage<Scalar, Size>>> : public std::true_type
{};

// Const map storage is ordered
template<typename Scalar, int Size>
struct is_ordered<const Eigen::Map<const DefaultStorage<Scalar, Size>>> : public std::true_type
{};

template<typename Storage>
static constexpr bool is_ordered_v = is_ordered<Storage>::value;

}  // namespace smooth

#endif  // SMOOTH__STORAGE_HPP_
