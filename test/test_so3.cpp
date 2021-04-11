#include <gtest/gtest.h>

#include "smooth/so3.hpp"
#include "smooth/map.hpp"


template<smooth::LieGroupLike T>
void test()
{}

TEST(SO3, LieGroupLike)
{
  test<smooth::SO3d>();
  test<smooth::Map<smooth::SO3d>>();
  test<smooth::ConstMap<smooth::SO3d>>();
}

TEST(SO3, Copy)
{
  smooth::SO3d so3;
  so3.setIdentity();

  std::array<double, 4> a1{0.5, 0.5, 0.5, 0.5}, a2{0, 0, 0, 1};
  smooth::Map<smooth::SO3d> m1(a1.data()), m2(a2.data());

  so3 = m1;  // map to so3
  m2 = m1;   // map to map

  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, so3.coeffs(), m1.coeffs());
  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, m2.coeffs(), m1.coeffs());

  so3.setIdentity();

  m1 = so3;  // so3 to map
  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, so3.coeffs(), m1.coeffs());
}

TEST(SO3, Constructors)
{
  auto q = Eigen::Quaternionf::UnitRandom();
  smooth::SO3<float> g(q);

  ASSERT_PRED2([](auto a, auto b) {return a.isApprox(b);}, g.coeffs(), q.coeffs());
}

TEST(SO3, CompositionInverse)
{
  const auto g_id = smooth::SO3d::Identity();
  ASSERT_TRUE((g_id * g_id).coeffs().isApprox(g_id.coeffs()));

  std::default_random_engine rng(5);
  for (auto i = 0u; i != 10; ++i) {
    const auto g = smooth::SO3d::Random(rng);
    const auto ginv = g.inverse();
    const auto g_ginv = g * ginv;
    ASSERT_TRUE(g_ginv.coeffs().isApprox(g_id.coeffs()));
  }
}

TEST(SO3, LogAndExp)
{
  // identity <-> zero
  auto g_id = smooth::SO3d::Identity();
  auto exp_0 = smooth::SO3d::exp(smooth::SO3d::Tangent::Zero());
  ASSERT_TRUE(g_id.coeffs().isApprox(exp_0.coeffs()));
  auto log_id = g_id.log();
  ASSERT_TRUE(log_id.isApprox(smooth::SO3d::Tangent::Zero()));

  // test some random ones
  std::default_random_engine rng(5);
  for (auto i = 0u; i != 10; ++i) {
    auto g = smooth::SO3d::Random(rng);
    auto log = g.log();
    auto g_copy = smooth::SO3d::exp(log);
    auto log_copy = g_copy.log();
    ASSERT_TRUE(g.coeffs().isApprox(g_copy.coeffs()));
    ASSERT_TRUE(log.isApprox(log_copy));
  }
}

TEST(SO3, Jacobians)
{
  // test zero vector
  auto dr_exp_0 = smooth::SO3d::dr_exp(smooth::SO3d::Tangent::Zero());
  auto dr_exp_inv_0 = smooth::SO3d::dr_exp(smooth::SO3d::Tangent::Zero());

  ASSERT_TRUE(dr_exp_0.isApprox(smooth::SO3d::TangentMap::Identity()));
  ASSERT_TRUE(dr_exp_inv_0.isApprox(smooth::SO3d::TangentMap::Identity()));

  std::default_random_engine rng(5);

  // check that they are each others inverses
  for (auto i = 0u; i != 10; ++i) {
    const smooth::SO3d::Tangent a = smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();
    const auto dr_exp_a = smooth::SO3d::dr_exp(a);
    const auto dr_expinv_a = smooth::SO3d::dr_expinv(a);

    const auto M1 = (dr_exp_a * dr_expinv_a).eval();
    const auto M2 = (dr_expinv_a * dr_exp_a).eval();

    ASSERT_TRUE(M1.isApprox(smooth::SO3d::TangentMap::Identity()));
    ASSERT_TRUE(M2.isApprox(smooth::SO3d::TangentMap::Identity()));
  }

  // check infnitesimal step for exp
  for (auto i = 0; i != 10; ++i) {
    const smooth::SO3d::Tangent a = smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const smooth::SO3d::Tangent da = 1e-4 * smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const auto dr_exp = smooth::SO3d::dr_exp(a);

    const auto g_exact = smooth::SO3d::exp(a + da);
    const auto g_approx = smooth::SO3d::exp(a) * smooth::SO3d::exp(dr_exp * da);

    ASSERT_TRUE(g_approx.coeffs().isApprox(g_exact.coeffs(), 1e-8));
  }

  // check infinitesimal step for log
  for (auto i = 0u; i != 10; ++i) {
    auto g = smooth::SO3d::Random(rng);
    const smooth::SO3d::Tangent dg = 1e-4 * smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const auto a_exact = (g * smooth::SO3d::exp(dg)).log();
    const auto a_approx = g.log() + smooth::SO3d::dr_expinv(g.log()) * dg;

    ASSERT_TRUE(a_exact.isApprox(a_approx, 1e-8));
  }
}
