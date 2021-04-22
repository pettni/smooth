#ifndef SMOOTH__STORAGE_HPP_
#define SMOOTH__STORAGE_HPP_

#include <Eigen/Core>

#include <cstdint>

#include "concepts.hpp"
#include "meta.hpp"


namespace smooth
{

template<typename _Scalar, std::size_t N>
class DefaultStorage
{
public:
  using Scalar = _Scalar;
  static constexpr uint32_t SizeAtCompileTime = N;

  Scalar & operator[](int i)
  {
    return a[i];
  }

  const Scalar & operator[](int i) const
  {
    return a[i];
  }

  Scalar * data()
  {
    return a.data();
  }

  const Scalar * data() const
  {
    return a.data();
  }

private:
  std::array<Scalar, N> a;
};


template<typename _Scalar, std::size_t N>
class MappedStorage
{
public:
  using Scalar = _Scalar;
  static constexpr uint32_t SizeAtCompileTime = N;

  MappedStorage(const Scalar * a_in)
  : a(a_in) {}
  // copy must copy underlying data
  MappedStorage(const MappedStorage & o) {memcpy(const_cast<Scalar *>(a), o.a, N * sizeof(_Scalar));}
  MappedStorage & operator=(const MappedStorage & o)
  {
    memcpy(const_cast<Scalar *>(a), o.a, N * sizeof(_Scalar)); return *this;
  }
  // for moving we move pointer
  MappedStorage(MappedStorage &&) = default;
  MappedStorage & operator=(MappedStorage &&) = default;
  ~MappedStorage() = default;

  Scalar & operator[](int i)
  {
    return const_cast<Scalar &>(a[i]);
  }

  const Scalar & operator[](int i) const
  {
    return a[i];
  }

  Scalar * data()
  {
    return const_cast<Scalar *>(a);
  }

  const Scalar * data() const
  {
    return a;
  }

private:
  const Scalar * a;
};


template<typename T>
struct map_trait;

/**
 * @brief Use base group with MappedStorage as map for lie groups
 */
template<LieGroupLike G>
struct map_trait<G>
{
  using type = meta::change_template_arg_t<
      meta::change_template_arg_t<G, 0, typename G::Scalar>,
      1, MappedStorage<typename G::Scalar, G::lie_size>
  >;

  using const_type = meta::change_template_arg_t<
    meta::change_template_arg_t<G, 0, typename G::Scalar>,
    1, const MappedStorage<typename G::Scalar, G::lie_size>
  >;
};

/**
 * @brief Use regular Eigen map as map for En
 */
template<EnLike G>
struct map_trait<G>
{
  using type = Eigen::Map<G>;
  using const_type = Eigen::Map<const G>;
};

/**
 * @brief Generic map type
 */
template<typename G>
using Map = typename map_trait<G>::type;

/**
 * @brief Generic const map type
 */
template<typename G>
using ConstMap = typename map_trait<G>::const_type;

}  // namespace smooth

#endif  // SMOOTH__STORAGE_HPP_
