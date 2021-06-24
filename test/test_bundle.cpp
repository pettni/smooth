#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/tn.hpp"


using namespace smooth;

TEST(Bundle, Static)
{
  using bundle_t = Bundle<SE2d, SE3d, T2d, SO2d, SO3d>;

  static_assert(bundle_t::RepSize == 19);
  static_assert(bundle_t::Dof == 15);
  static_assert(bundle_t::Dim == 15);

  static_assert(std::is_same_v<bundle_t::PartType<0>, SE2d>);
  static_assert(std::is_same_v<bundle_t::PartType<1>, SE3d>);
  static_assert(std::is_same_v<bundle_t::PartType<2>, Eigen::Vector2d>);
  static_assert(std::is_same_v<bundle_t::PartType<3>, SO2d>);
  static_assert(std::is_same_v<bundle_t::PartType<4>, SO3d>);
}

TEST(Bundle, Construct)
{
  std::srand(5);

  using mybundle = Bundle<SO2d, SO3d, T3d>;

  auto so2 = SO2d::Random();
  auto so3 = SO3d::Random();
  auto e3 = Eigen::Vector3d::Random().eval();

  mybundle b(so2, so3, e3);

  ASSERT_TRUE(b.part<0>().isApprox(so2));
  ASSERT_TRUE(b.part<1>().isApprox(so3));
  ASSERT_TRUE(b.part<2>().isApprox(e3));

  Eigen::Matrix4d m;
  m.setIdentity();
  m.topRightCorner<3, 1>() = e3;
  ASSERT_TRUE(m.isApprox(b.matrix().bottomRightCorner<4, 4>()));
}

using SubBundle = Bundle<SO3d, T3d>;

TEST(Bundle, BundleOfBundle)
{
  std::srand(5);

  using MetaBundle = Bundle<SO2d, SubBundle, SE2d>;

  auto so2 = SO2d::Random();
  auto so3 = SO3d::Random();
  auto e3 = Eigen::Vector3d::Random().eval();
  auto se2 = SE2d::Random();

  SubBundle sb(so3, e3);
  MetaBundle mb(std::move(so2), std::move(sb), std::move(se2));

  ASSERT_TRUE(mb.part<1>().part<0>().isApprox(so3));
  ASSERT_TRUE(mb.part<1>().part<1>().isApprox(e3));
}
