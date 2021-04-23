#ifndef SMOOTH__STORAGE_HPP_
#define SMOOTH__STORAGE_HPP_

#include <cstdint>

#include "concepts.hpp"


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
  MappedStorage(const MappedStorage & o)
  {
    memcpy(const_cast<Scalar *>(a), o.a, N * sizeof(_Scalar));
  }
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


/**
 * @brief Change storage type of a Lie Group type
 */
template<LieGroupLike G, typename NewStorage>
struct change_storage;

template<
  template<typename, typename, template<typename> typename ...> typename _G,
  typename _Scalar,
  typename _NewStorage,
  template<typename, std::size_t> typename _Storage,
  std::size_t lie_size,
  template<typename> typename ... _Ts
>
struct change_storage<_G<_Scalar, _Storage<_Scalar, lie_size>, _Ts...>, _NewStorage>
{
  using type = _G<_Scalar, _NewStorage, _Ts...>;
};

template<LieGroupLike G, typename NewScalar>
using change_storage_t = typename change_storage<G, NewScalar>::type;


template<typename T>
struct map_dispatcher;

/**
 * @brief Use base group with MappedStorage as map for lie groups
 */
template<LieGroupLike G>
struct map_dispatcher<G>
{
  using type = change_storage_t<G, MappedStorage<typename G::Scalar, G::lie_size>>;
};

/**
 * @brief Use base group with MappedStorage as map for lie groups
 */
template<LieGroupLike G>
struct map_dispatcher<const G>
{
  using type = change_storage_t<G, const MappedStorage<typename G::Scalar, G::lie_size>>;
};

/**
 * @brief Use regular Eigen map as map for En
 */
template<EnLike G>
struct map_dispatcher<G>
{
  using type = Eigen::Map<G>;
};

/**
 * @brief Use regular Eigen map as map for En
 */
template<EnLike G>
struct map_dispatcher<const G>
{
  using type = Eigen::Map<const G>;
};

/**
 * @brief Generic map type
 */
template<typename G>
using Map = typename map_dispatcher<G>::type;

}  // namespace smooth

#endif  // SMOOTH__STORAGE_HPP_
