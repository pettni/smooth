// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "../derivatives.hpp"

namespace smooth {
inline namespace v1_0 {

template<typename At, typename dAt, typename Bt, typename dBt>
auto d_matrix_product(const At & A, const dAt & dA, const Bt & B, const dBt & dB)
{
  using Scalar =
    std::common_type_t<typename At::Scalar, typename dAt::Scalar, typename Bt::Scalar, typename dBt::Scalar>;

  static constexpr int N    = At::ColsAtCompileTime;
  static constexpr int M    = Bt::RowsAtCompileTime;
  static constexpr int Nvar = []() -> int {
    if constexpr (dAt::ColsAtCompileTime > 0 && N > 0) {
      return dAt::ColsAtCompileTime / N;
    } else if (dBt::ColsAtCompileTime > 0 && M > 0) {
      return dBt::ColsAtCompileTime / M;
    } else {
      return -1;
    }
  }();

  const auto n                  = A.cols();
  [[maybe_unused]] const auto k = A.rows();
  const auto m                  = B.rows();
  const auto nvar               = dA.cols() / (n);

  assert(k == B.cols());
  assert(nvar == dB.size() / (m * k));

  static constexpr int dAB_cols = (M > 0 && Nvar > 0) ? M * Nvar : -1;

  Eigen::Matrix<Scalar, N, dAB_cols> dAB = B.transpose() * dA;
  for (auto i = 0u; i < n; ++i) {
    for (auto j = 0u; j < m; ++j) {
      dAB.template middleCols<Nvar>(i * nvar, nvar) += A(i, j) * dB.template middleCols<Nvar>(j * Nvar, nvar);
    }
  }
  return dAB;
}

template<typename JfT, typename HfT, typename JgT, typename HgT>
auto d2_fog(const JfT & Jf, const HfT & Hf, const JgT & Jg, const HgT & Hg)
{
  using Scalar =
    std::common_type_t<typename JfT::Scalar, typename HfT::Scalar, typename JgT::Scalar, typename HgT::Scalar>;

  static constexpr int No = JfT::RowsAtCompileTime;
  static constexpr int Ny = JfT::ColsAtCompileTime;
  static constexpr int Nx = JgT::ColsAtCompileTime;

  const auto no = Jf.rows();
  const auto ny = Jf.cols();

  [[maybe_unused]] const auto ni = Jg.rows();
  const auto nx                  = Jg.cols();

  // check some dimensions
  assert(ny == ni);
  assert(Hf.rows() == ny);
  assert(Hf.cols() == no * ny);
  assert(Hg.rows() == nx);
  assert(Hg.cols() == ni * nx);

  Eigen::Matrix<Scalar, Nx, (No == -1 || Nx == -1) ? -1 : No * Nx> ret(nx, no * nx);
  ret.setZero();

  for (auto i = 0u; i < no; ++i) {
    ret.template block<Nx, Nx>(0, i * nx, nx, nx) += Jg.transpose() * Hf.template middleCols<Ny>(i * ny, ny) * Jg;
  }

  for (auto i = 0u; i < Jf.outerSize(); ++i) {
    for (Eigen::InnerIterator it(Jf, i); it; ++it) {
      ret.template block<Nx, Nx>(0, it.row() * nx) += it.value() * Hg.template middleCols<Nx>(it.col() * nx, nx);
    }
  }

  return ret;
}

template<LieGroup G>
TangentMap<G> dr_rminus(const Tangent<G> & e)
{
  return dr_expinv<G>(e);
}
template<LieGroup G>
Hessian<G> d2r_rminus(const Tangent<G> & e)
{
  const auto J = dr_expinv<G>(e);

  auto res = d2r_expinv<G>(e);
  for (auto j = 0u; j < Dof<G>; ++j) {
    res.template block<Dof<G>, Dof<G>>(0, j * e.size(), e.size(), e.size()).applyOnTheRight(J);
  }
  return res;
}

template<LieGroup G>
Eigen::RowVector<Scalar<G>, Dof<G>> dr_rminus_squarednorm(const Tangent<G> & e)
{
  return e.transpose() * dr_expinv<G>(e);
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>> d2r_rminus_squarednorm(const Tangent<G> & e)
{
  const TangentMap<G> J1 = dr_rminus<G>(e);   // N x N
  const Hessian<G> H1    = d2r_rminus<G>(e);  // N x (N*N)

  return d2_fog(e.transpose(), Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>>::Identity(), J1, H1);
}

}  // namespace v1_0
}  // namespace smooth
