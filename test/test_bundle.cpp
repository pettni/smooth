#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"


using namespace smooth;

TEST(Bundle, iseq)
{
  using iseq_test = std::index_sequence<2, 4, 5, 1, 3>;
  using test_psum = iseq_psum<iseq_test>::type;

  static_assert(iseq_len<iseq_test>::value == 5);
  static_assert(iseq_el<2, iseq_test>::value == 5);
  static_assert(iseq_el<4, iseq_test>::value == 3);

  static_assert(std::is_same_v<test_psum, std::index_sequence<0, 2, 6, 11, 12>>);
}

TEST(Bundle, Static)
{
  using bundle_t = Bundle<double, SE2, SE3, SO2, SO3>;

  static_assert(bundle_t::lie_size == 17);
  static_assert(bundle_t::lie_dof == 13);
  static_assert(bundle_t::lie_dim == 12);
  static_assert(bundle_t::lie_actdim == 10);

  static_assert(std::is_same_v<bundle_t::PartType<0>, SE2d>);
  static_assert(std::is_same_v<bundle_t::PartType<1>, SE3d>);
  static_assert(std::is_same_v<bundle_t::PartType<2>, SO2d>);
  static_assert(std::is_same_v<bundle_t::PartType<3>, SO3d>);
}

template<typename Scalar>
using E3 = Eigen::Matrix<Scalar, 3, 1>;

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
