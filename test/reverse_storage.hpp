#ifndef TEST__REVERSE_STORAGE_HPP_
#define TEST__REVERSE_STORAGE_HPP_

#include <array>
#include <cstdint>

#include <Eigen/Core>

namespace smooth {

/**
 * @brief Reverse storage type for testing purposes
 */
template<typename _Scalar, uint32_t _Size>
struct ReverseStorage
{
  using Scalar                       = _Scalar;
  static constexpr Eigen::Index Size = _Size;

  ReverseStorage(Scalar * a) : a_(a) {}

  const Scalar & operator[](int i) const { return a_[Size - 1 - i]; }

  Scalar * a_;
};

}  // namespace smooth

#endif  // TEST__REVERSE_STORAGE_HPP_
