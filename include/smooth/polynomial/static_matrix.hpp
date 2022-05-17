// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Utilities for compile-time matrix algebra
 */

#include <array>
#include <cmath>

#include "smooth/detail/utils.hpp"

namespace smooth {

/**
 * @brief Elementary structure for compile-time matrix algebra.
 *
 * Matrix is stored in memory in row-major order.
 */
template<typename _Scalar, std::size_t _Rows, std::size_t _Cols>
struct StaticMatrix : public std::array<std::array<_Scalar, _Cols>, _Rows>
{
  /// @brief Number of rows in matrix
  static constexpr std::size_t Rows = _Rows;
  /// @brief Number of columns in matrix
  static constexpr std::size_t Cols = _Cols;

  using std::array<std::array<_Scalar, _Cols>, _Rows>::operator[];

  /**
   * @brief Construct matrix filled with zeros
   */
  constexpr StaticMatrix() : std::array<std::array<_Scalar, _Cols>, _Rows>{}
  {
    for (auto i = 0u; i != _Rows; ++i) { operator[](i).fill(_Scalar(0)); }
  }

  /**
   * @brief Extract sub-block of size _NRows x _NCols anchored at (row0, col0)
   */
  template<std::size_t _NRows, std::size_t _NCols>
  constexpr StaticMatrix<_Scalar, _NRows, _NCols> block(std::size_t row0, std::size_t col0) const
  {
    StaticMatrix<_Scalar, _NRows, _NCols> ret;
    for (auto i = 0u; i < _NRows; ++i) {
      for (auto j = 0u; j < _NCols; ++j) { ret[i][j] = operator[](row0 + i)[col0 + j]; }
    }
    return ret;
  }

  /**
   * @brief Matrix addition
   */
  constexpr StaticMatrix<_Scalar, _Rows, _Cols>
  operator+(const StaticMatrix<_Scalar, _Rows, _Cols> & o) const
  {
    StaticMatrix<_Scalar, _Rows, _Cols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _Cols; ++j) { ret[i][j] = operator[](i)[j] + o[i][j]; }
    }
    return ret;
  }

  /**
   * @brief Matrix transpose
   */
  constexpr StaticMatrix<_Scalar, _Rows, _Cols> transpose() const
  {
    StaticMatrix<_Scalar, _Rows, _Cols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _Cols; ++j) { ret[j][i] = operator[](i)[j]; }
    }
    return ret;
  }

  /**
   * @brief Matrix multiplication
   */
  template<std::size_t _NCols>
  constexpr StaticMatrix<_Scalar, _Rows, _NCols>
  operator*(const StaticMatrix<_Scalar, _Cols, _NCols> & o) const
  {
    StaticMatrix<_Scalar, _Rows, _NCols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _NCols; ++j) {
        for (auto k = 0u; k < _Cols; ++k) { ret[i][j] += operator[](i)[k] * o[k][j]; }
      }
    }
    return ret;
  }
};

}  // namespace smooth

