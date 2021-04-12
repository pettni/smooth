#include <gtest/gtest.h>

#include "smooth/so3.hpp"
#include "smooth/storage.hpp"


template<smooth::LieGroupLike T>
void test()
{}

TEST(SO3, Static)
{
  static_assert(smooth::SO3d::size == 4);
  static_assert(smooth::SO3d::dof == 3);
  static_assert(smooth::SO3d::dim == 3);

  static_assert(smooth::is_ordered<smooth::DefaultStorage<double, 4>>::value);
  static_assert(smooth::is_ordered<Eigen::Map<smooth::DefaultStorage<double, 4>>>::value);
  static_assert(smooth::is_ordered<const Eigen::Map<const smooth::DefaultStorage<double, 4>>>::value);
}

TEST(SO3, LieGroupLike)
{
  test<smooth::SO3d>();
  test<smooth::Map<smooth::SO3d>>();
  test<smooth::ConstMap<smooth::SO3d>>();
}

TEST(SO3, Constructors)
{
  std::default_random_engine rng(5);

  // un-initialized
  smooth::SO3d so3;
  so3.setRandom(rng);

  // map
  std::array<double, smooth::SO3d::size> a1;
  smooth::Map<smooth::SO3d> m1(a1.data());
  smooth::ConstMap<smooth::SO3d> m2(a1.data());
  m1.setRandom(rng);
  ASSERT_TRUE(m1.isApprox(m2));

  // copy constructor from group
  smooth::SO3d so3_copy(so3);
  ASSERT_TRUE(so3_copy.isApprox(so3));

  // copy constructor from map
  std::array<double, smooth::SO3d::size> a;
  smooth::Map<smooth::SO3d> m(a.data());
  m.setRandom(rng);
  smooth::SO3d m_copy(m);
  ASSERT_TRUE(m_copy.isApprox(m));

  // move contructor
  {
    smooth::SO3d g1, g2;
    g1.setRandom(rng);
    g2 = g1;
    smooth::SO3d g3(std::move(g1));
    ASSERT_TRUE(g3.isApprox(g2));
  }
}

TEST(SO3, Cast)
{
  std::default_random_engine rng(5);

  smooth::SO3d g;
  g.setRandom(rng);

  auto g_float = g.cast<float>();
  ASSERT_TRUE(g_float.log().isApprox(g.log().cast<float>()));
}

TEST(SO3, Copying)
{
  std::array<double, smooth::SO3d::size> a1, a2;
  smooth::SO3d g1, g2;
  smooth::Map<smooth::SO3d> m1(a1.data()), m2(a2.data());
  std::default_random_engine rng(5);

  // group to group
  g1.setRandom(rng);
  g2 = g1;
  ASSERT_TRUE(g2.isApprox(g1));

  // group to map
  g1.setRandom(rng);
  m1 = g1;
  ASSERT_TRUE(m1.isApprox(g1));

  // map to map
  m1.setRandom(rng);
  m2 = m1;
  ASSERT_TRUE(m2.isApprox(m1));

  // map to group
  m1.setRandom(rng);
  g1 = m1;
  ASSERT_TRUE(g1.isApprox(m1));

  // move group to group
  {
    smooth::SO3d g1, g2, g3;
    g1.setRandom(rng);
    g2 = g1;
    g3 = std::move(g1);
    ASSERT_TRUE(g3.isApprox(g2));
  }
}

TEST(SO3, CompositionInverse)
{
  const auto g_id = smooth::SO3d::Identity();
  ASSERT_TRUE((g_id * g_id).isApprox(g_id));

  std::default_random_engine rng(5);
  for (auto i = 0u; i != 10; ++i) {
    const auto g = smooth::SO3d::Random(rng);
    const auto ginv = g.inverse();
    const auto g_ginv = g * ginv;
    ASSERT_TRUE(g_ginv.isApprox(g_id));
  }
}

TEST(SO3, LogAndExp)
{
  // identity <-> zero
  auto g_id = smooth::SO3d::Identity();
  auto exp_0 = smooth::SO3d::exp(smooth::SO3d::Tangent::Zero());
  ASSERT_TRUE(g_id.isApprox(exp_0));
  auto log_id = g_id.log();
  ASSERT_TRUE(log_id.isApprox(smooth::SO3d::Tangent::Zero()));

  // test some random ones
  std::default_random_engine rng(5);
  for (auto i = 0u; i != 10; ++i) {
    auto g = smooth::SO3d::Random(rng);
    auto log = g.log();
    auto g_copy = smooth::SO3d::exp(log);
    auto log_copy = g_copy.log();
    ASSERT_TRUE(g.isApprox(g_copy));
    ASSERT_TRUE(log.isApprox(log_copy));
  }
}

TEST(SO3, Jacobians)
{
  // test zero vector
  const auto dr_exp_0 = smooth::SO3d::dr_exp(smooth::SO3d::Tangent::Zero());
  const auto dr_exp_inv_0 = smooth::SO3d::dr_exp(smooth::SO3d::Tangent::Zero());

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

  // check infinitesimal step for exp (right)
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

    ASSERT_TRUE(g_approx.isApprox(g_exact, 1e-8));
  }

  // check infinitesimal step for exp (left)
  for (auto i = 0; i != 10; ++i) {
    const smooth::SO3d::Tangent a = smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const smooth::SO3d::Tangent da = 1e-4 * smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const auto dl_exp = smooth::SO3d::dl_exp(a);

    const auto g_exact = smooth::SO3d::exp(da + a);
    const auto g_approx = smooth::SO3d::exp(dl_exp * da) * smooth::SO3d::exp(a);

    ASSERT_TRUE(g_approx.isApprox(g_exact, 1e-8));
  }

  // check infinitesimal step for log (right)
  for (auto i = 0u; i != 10; ++i) {
    auto g = smooth::SO3d::Random(rng);
    const smooth::SO3d::Tangent dg = 1e-4 * smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const auto a_exact = (g * smooth::SO3d::exp(dg)).log();
    const auto a_approx = g.log() + smooth::SO3d::dr_expinv(g.log()) * dg;

    ASSERT_TRUE(a_exact.isApprox(a_approx, 1e-8));
  }

  // check infinitesimal step for log (left)
  for (auto i = 0u; i != 10; ++i) {
    auto g = smooth::SO3d::Random(rng);
    const smooth::SO3d::Tangent dg = 1e-4 * smooth::SO3d::Tangent::NullaryExpr(
      [&rng] (int) {return smooth::u_distr<double>(rng);}
    ).eval();

    const auto a_exact = (smooth::SO3d::exp(dg) * g).log();
    const auto a_approx = smooth::SO3d::dl_expinv(g.log()) * dg + g.log();

    ASSERT_TRUE(a_exact.isApprox(a_approx, 1e-8));
  }
}
