#ifndef SMOOTH__STORAGE_HPP_
#define SMOOTH__STORAGE_HPP_

#include <Eigen/Core>

#include <cstdint>

#include "concepts.hpp"
#include "traits.hpp"


namespace smooth
{

template<LieGroupLike G>
struct map_trait
{
  using type = change_template_args_t<
    G,
    typename G::Scalar,
    Eigen::Map<DefaultStorage<typename G::Scalar, G::lie_size>>
  >;

  using const_type = change_template_args_t<
    G,
    typename G::Scalar,
    const Eigen::Map<const DefaultStorage<typename G::Scalar, G::lie_size>>
  >;
};

template<typename Scalar, int n>
struct map_trait<Eigen::Matrix<Scalar, n, 1>>
{
  using type = Eigen::Map<Eigen::Matrix<Scalar, n, 1>>;
  using const_type = Eigen::Map<const Eigen::Matrix<Scalar, n, 1>>;
};

/**
 * @brief Generic map type
 */
template<LieGroupLike G>
using Map = typename map_trait<G>::type;

/**
 * @brief Generic const map type
 */
template<LieGroupLike G>
using ConstMap = typename map_trait<G>::const_type;

}  // namespace smooth

#endif  // SMOOTH__STORAGE_HPP_
