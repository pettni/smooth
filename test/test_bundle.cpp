#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"


using namespace smooth;

TEST(Bundle, Static)
{
  using bundle_t = Bundle<double, SE2, SE3, E2, SO2, SO3>;

  static_assert(bundle_t::lie_size == 19);
  static_assert(bundle_t::lie_dof == 15);
  static_assert(bundle_t::lie_dim == 15);
  static_assert(bundle_t::lie_actdim == 12);

  static_assert(std::is_same_v<bundle_t::PartType<0>, SE2d>);
  static_assert(std::is_same_v<bundle_t::PartType<1>, SE3d>);
  static_assert(std::is_same_v<bundle_t::PartType<2>, Eigen::Vector2d>);
  static_assert(std::is_same_v<bundle_t::PartType<3>, SO2d>);
  static_assert(std::is_same_v<bundle_t::PartType<4>, SO3d>);
}

TEST(Bundle, Construct)
{
  std::default_random_engine rng(5);
  using mybundle = Bundle<double, SO2, SO3, E3>;

  auto so2 = SO2d::Random(rng);
  auto so3 = SO3d::Random(rng);
  auto e3 = Eigen::Vector3d::Random().eval();

  mybundle b(so2, so3, e3);

  ASSERT_TRUE(b.part<0>().isApprox(so2));
  ASSERT_TRUE(b.part<1>().isApprox(so3));
  ASSERT_TRUE(b.part<2>().isApprox(e3));

  Eigen::Matrix4d m;
  m.setIdentity();
  m.topRightCorner<3, 1>() = e3;
  ASSERT_TRUE(m.isApprox(b.matrix_group().bottomRightCorner<4, 4>()));
}

template<typename Scalar>
using SubBundle = Bundle<Scalar, SO3, E3>;

TEST(Bundle, BundleOfBundle)
{
  std::default_random_engine rng(5);
  using MetaBundle = Bundle<double, SO2, SubBundle, SE2>;

  auto so2 = SO2d::Random(rng);
  auto so3 = SO3d::Random(rng);
  auto e3 = Eigen::Vector3d::Random().eval();
  auto se2 = SE2d::Random(rng);

  SubBundle<double> sb(so3, e3);
  MetaBundle mb(std::move(so2), std::move(sb), std::move(se2));

  ASSERT_TRUE(mb.part<1>().part<0>().isApprox(so3));
  ASSERT_TRUE(mb.part<1>().part<1>().isApprox(e3));
}
