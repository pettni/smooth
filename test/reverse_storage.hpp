#ifndef TEST__REVERSE_STORAGE_HPP_
#define TEST__REVERSE_STORAGE_HPP_

#include <array>
#include <cstdint>


namespace smooth
{

/**
 * @brief Reverse storage type for testing purposes
 */
template<typename _Scalar, uint32_t _Size>
struct ReverseStorage
{
  using Scalar = _Scalar;
  static constexpr Eigen::Index Size = _Size;

  const Scalar & operator[](int i) const {
    return a[Size - 1 - i];
  }

  Scalar * a;
};

} // namespace smooth

#endif  // TEST__REVERSE_STORAGE_HPP_
