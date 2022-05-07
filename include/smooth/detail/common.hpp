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

#ifndef SMOOTH__INTERNAL__COMMON_HPP_
#define SMOOTH__INTERNAL__COMMON_HPP_

namespace smooth {

static constexpr double eps2 = 1e-8;

#define SMOOTH_DEFINE_REFS                                                      \
  using GRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, RepSize, 1>> &;  \
  using GRefOut = Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>>;                \
                                                                                \
  using TRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, 1>> &;      \
  using TRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, 1>>;                    \
                                                                                \
  using TMapRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, Dof>> &; \
  using TMapRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof>>;               \
                                                                                \
  using THessRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof * Dof>>;        \
                                                                                \
  using MRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dim, Dim>> &;    \
  using MRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>;                  \
                                                                                \
  static_assert(true)

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__COMMON_HPP_
