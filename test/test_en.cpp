#include <gtest/gtest.h>

#include <smooth/en.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>

TEST(En, Basic)
{
  std::default_random_engine rng(5);

  static_assert(std::is_same_v<Eigen::Vector3d::Group, Eigen::Vector3d>);
  static_assert(std::is_same_v<Eigen::Vector3d::MatrixGroup, Eigen::Matrix4d>);
  static_assert(std::is_same_v<Eigen::Vector3d::Tangent, Eigen::Vector3d>);
  static_assert(std::is_same_v<Eigen::Vector3d::TangentMap, Eigen::Matrix3d>);
  static_assert(std::is_same_v<Eigen::Vector3d::Vector, Eigen::Vector3d>);

  Eigen::Vector3d v;
  v.setIdentity();
  ASSERT_TRUE(v.isApprox(Eigen::Vector3d::Zero()));

  ASSERT_TRUE((v * v).isApprox(2 * v));
  ASSERT_TRUE(v.inverse().isApprox(-v));

  ASSERT_TRUE(v.log().isApprox(v));
  ASSERT_TRUE(Eigen::Vector3d::exp(v).isApprox(v));

  Eigen::Matrix3d M;
  M.setZero();
  const auto Mexp = M.exp();
  ASSERT_TRUE(Mexp.isApprox(Eigen::Matrix3d::Identity()));

  auto v_rand = Eigen::Vector3d::Random(rng);
  static_cast<void>(v_rand);

}
