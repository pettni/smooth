#include <gtest/gtest.h>

#include "smooth/interp/bspline.hpp"
#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"


TEST(BSpline, Static)
{
  constexpr auto c3 = smooth::detail::cum_card_coeffmat<double, 3>();
  static_assert(std::abs(c3[0][0] - 1) < 1e-8);
  static_assert(std::abs(c3[0][1] - 5. / 6) < 1e-8);
  static_assert(std::abs(c3[0][2] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[0][3] - 0) < 1e-8);

  static_assert(std::abs(c3[1][0] - 0) < 1e-8);
  static_assert(std::abs(c3[1][1] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][3] - 0) < 1e-8);

  static_assert(std::abs(c3[2][0] - 0) < 1e-8);
  static_assert(std::abs(c3[2][1] - -3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][3] - 0) < 1e-8);

  static_assert(std::abs(c3[3][0] - 0) < 1e-8);
  static_assert(std::abs(c3[3][1] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[3][2] - -2. / 6) < 1e-8);
  static_assert(std::abs(c3[3][3] - 1. / 6) < 1e-8);
}


template<smooth::LieGroupLike G>
class BSpline : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  smooth::SO2d, smooth::SO3d, smooth::SE2d, smooth::SE3d,
  smooth::Bundle<double, smooth::SO3, smooth::E4, smooth::SE2>
>;

TYPED_TEST_SUITE(BSpline, GroupsToTest);


TYPED_TEST(BSpline, Constant)
{
  std::default_random_engine rng(5);

  smooth::meta::static_for<6>(
    [&rng](auto k) {
      static constexpr uint32_t K = k + 1;

      std::vector<TypeParam> ctrl_pts;
      ctrl_pts.push_back(TypeParam::Random(rng));
      for (auto i = 0u; i != K; ++i) {
        ctrl_pts.push_back(ctrl_pts.back());
      }

      for (double u = 0.; u < 1; u += 0.05) {
        typename TypeParam::Tangent vel, acc;
        auto g = smooth::bspline_eval<TypeParam, K>(ctrl_pts, u, vel, acc);

        ASSERT_TRUE(g.isApprox(ctrl_pts.front()));
        ASSERT_TRUE(vel.norm() <= 1e-8);
        ASSERT_TRUE(acc.norm() <= 1e-8);
      }
    });
}


TEST(Smooth, BSpline)
{
  std::default_random_engine rng(5);

  std::vector<smooth::SO3d> c;
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));
  c.push_back(smooth::SO3d::Random(rng));

  typename smooth::SO3d::Tangent vel, acc;

  auto G = smooth::bspline_eval<smooth::SO3d, 3>(c, 0.5, vel, acc);

  std::cout << G << std::endl;
  std::cout << vel << std::endl;
  std::cout << acc << std::endl;
}
