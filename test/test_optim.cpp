#include <gtest/gtest.h>

#include "smooth/optim/lm.hpp"

template <int N, int M>
void run_leastsquares_test(bool zero_d, bool sing)
{
  // static
  for (auto i = 0u; i != 10; ++i) {
    Eigen::Matrix<double, M, N> J;
    Eigen::Matrix<double, M, 1> r;
    Eigen::Matrix<double, N, 1> d;

    J.setRandom();
    if (sing) {
      J.col(N / 2).setZero();
      J.row(M / 2).setZero();
    }

    d.setRandom();
    d = (d + Eigen::Matrix<double, N, 1>::Ones()).cwiseMax(0);
    if (zero_d) {
      d.setZero();
    }

    r.setRandom();

    // solve static
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto a1 = solve_ls<N, M>(J_qr, d, r);

    // solve dynamic
    Eigen::Matrix<double, -1, -1> Jd = J;
    Eigen::Matrix<double, -1, 1> rd = r;
    Eigen::Matrix<double, -1, 1> dd = d;
    Eigen::ColPivHouseholderQR<decltype(Jd)> Jd_qr(Jd);
    auto a2 = solve_ls<-1, -1>(Jd_qr, dd, rd);

    // verify solution
    Eigen::Matrix<double, N + M, N> lhs;
    lhs.template topLeftCorner<M, N>() = J;
    lhs.template bottomLeftCorner<N, N>() = d.asDiagonal();

    Eigen::Matrix<double, N + M, 1> rhs;
    rhs.template head<M>() = -r;
    rhs.template tail<N>().setZero();
    Eigen::Matrix<double, N, 1> a_verif = lhs.fullPivHouseholderQr().solve(rhs);

    ASSERT_TRUE(a1.isApprox(a_verif));
    ASSERT_TRUE(a2.isApprox(a_verif));
  }
}

TEST(Optimization, LeastSquares)
{
  run_leastsquares_test<1, 1>(false, false);
  run_leastsquares_test<5, 1>(false, false);
  run_leastsquares_test<5, 10>(false, false);
  run_leastsquares_test<8, 16>(false, false);

  run_leastsquares_test<1, 1>(false, true);
  run_leastsquares_test<5, 1>(false, true);
  run_leastsquares_test<5, 10>(false, true);
  run_leastsquares_test<8, 16>(false, true);

  run_leastsquares_test<1, 1>(true, false);
  run_leastsquares_test<5, 1>(true, false);
  run_leastsquares_test<5, 10>(true, false);
  run_leastsquares_test<8, 16>(true, false);

  run_leastsquares_test<1, 1>(true, true);
  run_leastsquares_test<5, 1>(true, true);
  run_leastsquares_test<5, 10>(true, true);
  run_leastsquares_test<8, 16>(true, true);
}

TEST(Optimization, LmPar)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 1;

  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    Eigen::Matrix<double, N, 1> x;

    // solve QR
    auto par = lmpar<4, 4>(J, d, r, Delta, x);

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = solve_ls<N, N>(J_qr, sqrt(par) * d, r);
    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);
  }
}

TEST(Optimization, LmParSmall)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 0.1;
  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    Eigen::Matrix<double, N, 1> x;

    // solve static
    auto par1 = lmpar<4, 4>(J, d, r, Delta, x);

    // solve dynamic
    Eigen::MatrixXd Jd = J;
    Eigen::VectorXd rd = r;
    Eigen::VectorXd dd = d;
    Eigen::VectorXd xd(N);
    auto par2 = lmpar<-1, -1>(Jd, dd, rd, Delta, xd);

    // check equality of static and dynamic
    ASSERT_EQ(par1, par2);
    ASSERT_TRUE(x.isApprox(xd));

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = solve_ls<N, N>(J_qr, sqrt(par1) * d, r);
    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par1 == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par1 > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);
  }
}

TEST(Optimization, LmParSing)
{
  constexpr int M = 4, N = 4;
  Eigen::Matrix<double, M, N> J;
  Eigen::Matrix<double, M, 1> r;
  Eigen::Matrix<double, N, 1> d;

  double Delta = 1;
  for (auto i = 0u; i != 10; ++i) {
    J.setRandom();
    J.col(3).setZero();

    d.setRandom();
    d = d + Eigen::Matrix<double, N, 1>::Ones();

    r.setRandom();

    Eigen::Matrix<double, N, 1> x;

    // solve QR
    auto par1 = lmpar<4, 4>(J, d, r, Delta, x);

    // solve dynamic
    Eigen::MatrixXd Jd = J;
    Eigen::VectorXd rd = r;
    Eigen::VectorXd dd = d;
    Eigen::VectorXd xd(N);
    auto par2 = lmpar<-1, -1>(Jd, dd, rd, Delta, xd);

    // check equality of static and dynamic
    ASSERT_EQ(par1, par2);
    ASSERT_TRUE(x.isApprox(xd));

    // check that x solves resulting problem
    Eigen::ColPivHouseholderQR<decltype(J)> J_qr(J);
    auto x_test = solve_ls<N, N>(J_qr, sqrt(par1) * d, r);

    ASSERT_TRUE(x_test.isApprox(x));

    // check that parameter satisfies conditions
    bool cond1 = (par1 == 0) && (d.asDiagonal() * x).norm() <= 1.1 * Delta;
    bool cond2 = (par1 > 0) && std::abs((d.asDiagonal() * x).norm() - Delta) <= 0.1 * Delta;
    ASSERT_TRUE(cond1 || cond2);
  }
}
