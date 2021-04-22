#ifndef SMOOTH__COMPAT__ROS_HPP_
#define SMOOTH__COMPAT__ROS_HPP_

#include <geometry_msgs/msg/pose.hpp>

#include "smooth/storage.hpp"
#include "smooth/se3.hpp"


namespace smooth
{

class PoseStorage : public MappedStorage<double, 7>
{
public:
  using Scalar = double;
  static constexpr uint32_t SizeAtCompileTime = 7;

  // statically check that what we map is in the expected places
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::position.x), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::position.y), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::position.z), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::orientation.x), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::orientation.y), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::orientation.z), double>);
  static_assert(std::is_same_v<decltype(geometry_msgs::msg::Pose::orientation.w), double>);
  static_assert(offsetof(geometry_msgs::msg::Pose, position.x) == 0);
  static_assert(offsetof(geometry_msgs::msg::Pose, position.y) == sizeof(double));
  static_assert(offsetof(geometry_msgs::msg::Pose, position.z) == 2 * sizeof(double));
  static_assert(offsetof(geometry_msgs::msg::Pose, orientation.x) == 3 * sizeof(double));
  static_assert(offsetof(geometry_msgs::msg::Pose, orientation.y) == 4 * sizeof(double));
  static_assert(offsetof(geometry_msgs::msg::Pose, orientation.z) == 5 * sizeof(double));
  static_assert(offsetof(geometry_msgs::msg::Pose, orientation.w) == 6 * sizeof(double));

  PoseStorage(const geometry_msgs::msg::Pose & p)
  : MappedStorage(reinterpret_cast<const double *>(&p))
  {}
};

template<>
struct map_trait<::geometry_msgs::msg::Pose>
{
  using type = smooth::SE3<double, PoseStorage>;
};

template<>
struct map_trait<const ::geometry_msgs::msg::Pose>
{
  using type = smooth::SE3<double, const PoseStorage>;
};

}   // namespace smooth

#endif  // SMOOTH__COMPAT__ROS_HPP_
