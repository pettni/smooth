#ifndef SMOOTH__MAP_HPP_
#define SMOOTH__MAP_HPP_

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

}  // namespace smooth

#endif  // SMOOTH__MAP_HPP_
