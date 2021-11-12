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

#ifndef SMOOTH__COMPAT__ROS_HPP_
#define SMOOTH__COMPAT__ROS_HPP_

/**
 * @file
 * @brief ROS message compatablity header.
 */

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>

#include "smooth/se3.hpp"

using namespace geometry_msgs::msg;

// Statically check that messages are laid out as expected in memory

// Vector3
static_assert(std::is_same_v<decltype(Vector3::x), double>);
static_assert(std::is_same_v<decltype(Vector3::y), double>);
static_assert(std::is_same_v<decltype(Vector3::z), double>);

static_assert(offsetof(Vector3, x) == 0);
static_assert(offsetof(Vector3, y) == sizeof(double));
static_assert(offsetof(Vector3, z) == 2 * sizeof(double));

// Point
static_assert(std::is_same_v<decltype(Point::x), double>);
static_assert(std::is_same_v<decltype(Point::y), double>);
static_assert(std::is_same_v<decltype(Point::z), double>);

static_assert(offsetof(Point, x) == 0);
static_assert(offsetof(Point, y) == sizeof(double));
static_assert(offsetof(Point, z) == 2 * sizeof(double));

// Quaternion
static_assert(std::is_same_v<decltype(Quaternion::x), double>);
static_assert(std::is_same_v<decltype(Quaternion::y), double>);
static_assert(std::is_same_v<decltype(Quaternion::z), double>);
static_assert(std::is_same_v<decltype(Quaternion::w), double>);

static_assert(offsetof(Quaternion, x) == 0);
static_assert(offsetof(Quaternion, y) == sizeof(double));
static_assert(offsetof(Quaternion, z) == 2 * sizeof(double));
static_assert(offsetof(Quaternion, w) == 3 * sizeof(double));

// Pose
static_assert(std::is_same_v<decltype(Pose::position), Point>);
static_assert(std::is_same_v<decltype(Pose::orientation), Quaternion>);
static_assert(offsetof(Pose, position) == 0);
static_assert(offsetof(Pose, orientation) == sizeof(Point));

// Tranform
static_assert(std::is_same_v<decltype(Transform::translation), Vector3>);
static_assert(std::is_same_v<decltype(Transform::rotation), Quaternion>);
static_assert(offsetof(Transform, translation) == 0);
static_assert(offsetof(Transform, rotation) == sizeof(Vector3));

// generic

//! Map message DATATYPE as implementation LIETYPE with CRTP base BASETYPE
#define CREATE_MAPS(DATATYPE, LIETYPE, BASETYPE)                                      \
  /*! @brief Specialize liebase_info       . */                                         \
  template<>                                                                          \
  struct smooth::lie_traitslie_traits<smooth::Map<DATATYPE>> : public lie_traitslie_traits<LIETYPE>       \
  {};                                                                                 \
                                                                                      \
  /*! @brief Memory mapping of ROS message as Lie group type. */                      \
  template<>                                                                          \
  class smooth::Map<DATATYPE> : public BASETYPE<smooth::Map<DATATYPE>>                \
  {                                                                                   \
    using Base = BASETYPE<smooth::Map<DATATYPE>>;                                     \
                                                                                      \
  public:                                                                             \
    /*! @brief Define types. */                                                       \
    SMOOTH_INHERIT_TYPEDEFS;                                                          \
                                                                                      \
    /*! Map message as Lie group type. */                                             \
    Map(DATATYPE & msg) : coeffs_(reinterpret_cast<double *>(&msg)) {}                \
    /*! Underlying storage is Eigen::Map */                                           \
    using Storage = Eigen::Map<Eigen::Matrix<double, RepSize, 1>>;                    \
    /*! Access underlying Eigen::Map */                                               \
    Storage & coeffs() { return coeffs_; }                                            \
    /*! Const access underlying Eigen::Map */                                         \
    const Storage & coeffs() const { return coeffs_; }                                \
    /*! Access raw pointer */                                                         \
    Scalar * data() { return coeffs_.data(); }                                        \
    /*! Const access raw pointer */                                                   \
    const Scalar * data() const { return coeffs_.data(); }                            \
                                                                                      \
  private:                                                                            \
    Storage coeffs_;                                                                  \
  };                                                                                  \
                                                                                      \
  /*! @brief Specialize liebase_info       . */                                         \
  template<>                                                                          \
  struct smooth::liebase_info<smooth::Map<const DATATYPE>> : public liebase_info<LIETYPE> \
  {                                                                                   \
    /*! @brief Const mapping is not mutable. */                                       \
    static constexpr bool is_mutable = false;                                         \
  };                                                                                  \
                                                                                      \
  /*! @brief Const memory mapping of ROS message as Lie group type. */                \
  template<>                                                                          \
  class smooth::Map<const DATATYPE> : public BASETYPE<smooth::Map<const DATATYPE>>    \
  {                                                                                   \
    using Base = BASETYPE<smooth::Map<const DATATYPE>>;                               \
                                                                                      \
  public:                                                                             \
    /*! @brief Define types. */                                                       \
    SMOOTH_INHERIT_TYPEDEFS;                                                          \
                                                                                      \
    /*! Const map message as Lie group type. */                                       \
    Map(const DATATYPE & msg) : coeffs_(reinterpret_cast<const double *>(&msg)) {}    \
    /*! Underlying storage is Eigen const Map */                                      \
    using Storage = Eigen::Map<const Eigen::Matrix<double, RepSize, 1>>;              \
    /*! Access underlying Eigen::Map */                                               \
    const Storage & coeffs() const { return coeffs_; }                                \
    /*! Access raw pointer */                                                         \
    const Scalar * data() const { return coeffs_.data(); }                            \
                                                                                      \
  private:                                                                            \
    Storage coeffs_;                                                                  \
  };                                                                                  \
                                                                                      \
  static_assert(true, "")


CREATE_MAPS(geometry_msgs::msg::Quaternion, smooth::SO3d, smooth::SO3Base);
CREATE_MAPS(geometry_msgs::msg::Pose, smooth::SE3d, smooth::SE3Base);
CREATE_MAPS(geometry_msgs::msg::Transform, smooth::SE3d, smooth::SE3Base);

#endif  // SMOOTH__COMPAT__ROS_HPP_
