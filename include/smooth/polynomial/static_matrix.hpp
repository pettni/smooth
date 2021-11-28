// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__POLYNOMIAL__STATIC_MATRIX_HPP_
#define SMOOTH__POLYNOMIAL__STATIC_MATRIX_HPP_

/**
 * @file
 * @brief Utilities for compile-time matrix algebra
 */

#include <array>
#include <cmath>

#include "smooth/internal/utils.hpp"

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

#endif  // SMOOTH__POLYNOMIAL__STATIC_MATRIX_HPP_
