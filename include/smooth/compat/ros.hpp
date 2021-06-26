#ifndef SMOOTH__COMPAT__ROS_HPP_
#define SMOOTH__COMPAT__ROS_HPP_

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

#define CREATE_MAPS(DATATYPE, LIETYPE, BASETYPE)                                     \
  template<>                                                                         \
  struct smooth::lie_traits<Eigen::Map<DATATYPE>> : public lie_traits<LIETYPE>       \
  {};                                                                                \
                                                                                     \
  template<>                                                                         \
  class Eigen::Map<DATATYPE> : public BASETYPE<Eigen::Map<DATATYPE>>                 \
  {                                                                                  \
    using Base = BASETYPE<Eigen::Map<DATATYPE>>;                                     \
                                                                                     \
  public:                                                                            \
    SMOOTH_INHERIT_TYPEDEFS;                                                         \
    Map(DATATYPE & msg) : coeffs_(reinterpret_cast<double *>(&msg)) {}               \
    using Storage = Eigen::Map<Eigen::Matrix<double, RepSize, 1>>;                   \
    Storage & coeffs() { return coeffs_; }                                           \
    const Storage & coeffs() const { return coeffs_; }                               \
    Scalar * data() { return coeffs_.data(); }                                       \
    const Scalar * data() const { return coeffs_.data(); }                           \
                                                                                     \
  private:                                                                           \
    Storage coeffs_;                                                                 \
  };                                                                                 \
                                                                                     \
  template<>                                                                         \
  struct smooth::lie_traits<Eigen::Map<const DATATYPE>> : public lie_traits<LIETYPE> \
  {                                                                                  \
    static constexpr bool is_mutable = false;                                        \
  };                                                                                 \
                                                                                     \
  template<>                                                                         \
  class Eigen::Map<const DATATYPE> : public BASETYPE<Eigen::Map<const DATATYPE>>     \
  {                                                                                  \
    using Base = BASETYPE<Eigen::Map<const DATATYPE>>;                               \
                                                                                     \
  public:                                                                            \
    SMOOTH_INHERIT_TYPEDEFS;                                                         \
    Map(const DATATYPE & msg) : coeffs_(reinterpret_cast<const double *>(&msg)) {}   \
    using Storage = Eigen::Map<const Eigen::Matrix<double, RepSize, 1>>;             \
    const Storage & coeffs() const { return coeffs_; }                               \
    const Scalar * data() const { return coeffs_.data(); }                           \
                                                                                     \
  private:                                                                           \
    Storage coeffs_;                                                                 \
  };                                                                                 \
                                                                                     \
  static_assert(true, "")

CREATE_MAPS(geometry_msgs::msg::Quaternion, smooth::SO3d, smooth::SO3Base);
CREATE_MAPS(geometry_msgs::msg::Pose, smooth::SE3d, smooth::SE3Base);
CREATE_MAPS(geometry_msgs::msg::Transform, smooth::SE3d, smooth::SE3Base);

#endif  // SMOOTH__COMPAT__ROS_HPP_
