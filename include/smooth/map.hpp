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

#ifndef SMOOTH__MAP_HPP_
#define SMOOTH__MAP_HPP_

#include <Eigen/Core>

#include "lie_group.hpp"

namespace smooth {

/**
 * @brief Memory mapping of internal Lie group types.
 */
template<typename T>
class Map;

// \cond
template<typename T>
struct map_traits;

template<NativeLieGroup G>
struct map_traits<G>
{
  using type = ::smooth::Map<G>;
};

template<NativeLieGroup G>
struct map_traits<const G>
{
  using type = ::smooth::Map<const G>;
};

template<typename _Scalar, int _N>
struct map_traits<Eigen::Matrix<_Scalar, _N, 1>>
{
  using type = ::Eigen::Map<Eigen::Matrix<_Scalar, _N, 1>>;
};

template<typename _Scalar, int _N>
struct map_traits<const Eigen::Matrix<_Scalar, _N, 1>>
{
  using type = ::Eigen::Map<const Eigen::Matrix<_Scalar, _N, 1>>;
};
// \endcond

/**
 * @brief Send smooth types to smooth::Map and Eigen types to Eigen::Map.
 */
template<typename T>
using MapDispatch = typename map_traits<T>::type;

}  // namespace smooth

#endif  // SMOOTH__MAP_HPP_
