#ifndef TEST__REVERSE_STORAGE_HPP_
#define TEST__REVERSE_STORAGE_HPP_

#include <array>
#include <cstdint>

namespace smooth
{

/**
 * @brief Reverse storage type for testing purposes
 */
template<typename _Scalar, uint32_t size>
struct ReverseStorage
{
  using Scalar = _Scalar;
  static constexpr uint32_t SizeAtCompileTime = size;

  Scalar & operator[](int i) {
    return a[size - 1 - i];
  }

  const Scalar & operator[](int i) const {
    return a[size - 1 - i];
  }

  std::array<Scalar, size> a;
};

} // namespace smooth

#endif  // TEST__REVERSE_STORAGE_HPP_