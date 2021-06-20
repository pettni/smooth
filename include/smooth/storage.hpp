#ifndef SMOOTH__STORAGE_HPP_
#define SMOOTH__STORAGE_HPP_

#include <cstdint>

#include "concepts.hpp"

namespace smooth {

template<typename _Scalar, std::size_t N>
class DefaultStorage {
public:
  using Scalar                       = _Scalar;
  static constexpr Eigen::Index Size = N;

  Scalar & operator[](int i) { return a[i]; }
  const Scalar & operator[](int i) const { return a[i]; }
  Scalar * data() { return a.data(); }
  const Scalar * data() const { return a.data(); }

private:
  std::array<Scalar, N> a;
};

template<typename _Scalar, std::size_t N>
class MappedStorage {
public:
  using Scalar                       = _Scalar;
  static constexpr Eigen::Index Size = N;

  MappedStorage(Scalar * a_in) : a(a_in) {}

  // copy construction is not allowed
  MappedStorage(const MappedStorage & o) = delete;

  // copy must copy underlying data
  MappedStorage & operator=(const MappedStorage & o)
  {
    memcpy(a, o.a, N * sizeof(_Scalar));
    return *this;
  }

  // for moving we move pointer
  MappedStorage(MappedStorage &&)             = default;
  MappedStorage & operator=(MappedStorage &&) = default;

  // memory is not managed
  ~MappedStorage() = default;

  Scalar & operator[](int i) { return a[i]; }
  const Scalar & operator[](int i) const { return a[i]; }
  Scalar * data() { return a; }
  const Scalar * data() const { return a; }

private:
  Scalar * const a;
};

template<typename _Scalar, std::size_t N>
class ConstMappedStorage {
public:
  using Scalar                       = _Scalar;
  static constexpr Eigen::Index Size = N;

  ConstMappedStorage(const Scalar * a_in) : a(a_in) {}

  // can not modify
  ConstMappedStorage(const ConstMappedStorage & o) = delete;
  ConstMappedStorage & operator=(const ConstMappedStorage & o) = delete;

  // for moving we move pointer
  ConstMappedStorage(ConstMappedStorage &&)             = default;
  ConstMappedStorage & operator=(ConstMappedStorage &&) = default;

  // memory is not managed
  ~ConstMappedStorage()                            = default;

  const Scalar & operator[](int i) const { return a[i]; }
  const Scalar * data() const { return a; }

private:
  const Scalar * const a;
};


template<typename T>
struct map_dispatcher;

/**
 * @brief Use base group with MappedStorage as map for lie groups
 */
template<LieGroup G>
struct map_dispatcher<G>
{
  using type = typename G::template NewStorageType<MappedStorage<typename G::Scalar, G::RepSize>>;
};

/**
 * @brief Use base group with MappedStorage as map for lie groups
 */
template<LieGroup G>
struct map_dispatcher<const G>
{
  using type = typename G::template NewStorageType<ConstMappedStorage<typename G::Scalar, G::RepSize>>;
};

/**
 * @brief Use regular Eigen map as map for En
 */
template<RnLike G>
struct map_dispatcher<G>
{
  using type = Eigen::Map<G>;
};

/**
 * @brief Use regular Eigen map as map for En
 */
template<RnLike G>
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
