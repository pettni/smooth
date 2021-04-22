#ifndef SMOOTH__COMPAT__ROS_HPP_
#define SMOOTH__COMPAT__ROS_HPP_

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>

#include "smooth/storage.hpp"
#include "smooth/se3.hpp"

// Statically check that messages are laid out as expected in memory

// Vector3
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Vector3::x), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Vector3::y), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Vector3::z), double>);

static_assert(offsetof(geometry_msgs::msg::Vector3, x) == 0);
static_assert(offsetof(geometry_msgs::msg::Vector3, y) == sizeof(double));
static_assert(offsetof(geometry_msgs::msg::Vector3, z) == 2 * sizeof(double));

// Point
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Point::x), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Point::y), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Point::z), double>);

static_assert(offsetof(geometry_msgs::msg::Point, x) == 0);
static_assert(offsetof(geometry_msgs::msg::Point, y) == sizeof(double));
static_assert(offsetof(geometry_msgs::msg::Point, z) == 2 * sizeof(double));

// Quaternion
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Quaternion::x), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Quaternion::y), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Quaternion::z), double>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Quaternion::w), double>);

static_assert(offsetof(geometry_msgs::msg::Quaternion, x) == 0);
static_assert(offsetof(geometry_msgs::msg::Quaternion, y) == sizeof(double));
static_assert(offsetof(geometry_msgs::msg::Quaternion, z) == 2 * sizeof(double));
static_assert(offsetof(geometry_msgs::msg::Quaternion, w) == 3 * sizeof(double));

// Pose
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::position), geometry_msgs::msg::Point>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::orientation), geometry_msgs::msg::Quaternion>);
static_assert(offsetof(geometry_msgs::msg::Pose, position) == 0);
static_assert(offsetof(geometry_msgs::msg::Pose, orientation) == sizeof(geometry_msgs::msg::Point));

// Tranform
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Transform::translation), geometry_msgs::msg::Vector3>);
static_assert(std::is_same_v<decltype(geometry_msgs::msg::Transform::rotation), geometry_msgs::msg::Quaternion>);
static_assert(offsetof(geometry_msgs::msg::Transform, translation) == 0);
static_assert(offsetof( geometry_msgs::msg::Transform, rotation) == sizeof(geometry_msgs::msg::Vector3));


namespace smooth
{

template<typename MsgType>
class RosMsgStorage : public MappedStorage<double, sizeof(MsgType) / sizeof(double)>
{
public:
  using Scalar = double;
  static constexpr uint32_t SizeAtCompileTime = sizeof(MsgType) / sizeof(double);

  RosMsgStorage(const MsgType & p)
  : MappedStorage<double, SizeAtCompileTime>(reinterpret_cast<const double *>(&p))
  {}
};

// Vector3
template<>
struct map_dispatcher<::geometry_msgs::msg::Vector3>
{
  using type = Eigen::Map<Eigen::Vector3d>;
};
template<>
struct map_dispatcher<const ::geometry_msgs::msg::Vector3>
{
  using type = Eigen::Map<const Eigen::Vector3d>;
};

// Point
template<>
struct map_dispatcher<::geometry_msgs::msg::Point>
{
  using type = Eigen::Map<Eigen::Vector3d>;
};
template<>
struct map_dispatcher<const ::geometry_msgs::msg::Point>
{
  using type = Eigen::Map<const Eigen::Vector3d>;
};

// Quaternion
template<>
struct map_dispatcher<::geometry_msgs::msg::Quaternion>
{
  using type = smooth::SO3<double, RosMsgStorage<::geometry_msgs::msg::Quaternion>>;
};
template<>
struct map_dispatcher<const ::geometry_msgs::msg::Quaternion>
{
  using type = smooth::SO3<double, const RosMsgStorage<::geometry_msgs::msg::Quaternion>>;
};

// Pose
template<>
struct map_dispatcher<::geometry_msgs::msg::Pose>
{
  using type = smooth::SE3<double, RosMsgStorage<::geometry_msgs::msg::Pose>>;
};
template<>
struct map_dispatcher<const ::geometry_msgs::msg::Pose>
{
  using type = smooth::SE3<double, const RosMsgStorage<::geometry_msgs::msg::Pose>>;
};

// Transform
template<>
struct map_dispatcher<::geometry_msgs::msg::Transform>
{
  using type = smooth::SE3<double, RosMsgStorage<::geometry_msgs::msg::Transform>>;
};
template<>
struct map_dispatcher<const ::geometry_msgs::msg::Transform>
{
  using type = smooth::SE3<double, const RosMsgStorage<::geometry_msgs::msg::Transform>>;
};

}   // namespace smooth

#endif  // SMOOTH__COMPAT__ROS_HPP_
