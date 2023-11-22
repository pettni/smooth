// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "smooth/version.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief Unrolled integer exponential.
 */
template<std::size_t e>
auto fpow(auto x) -> decltype(x)
{
  const auto f_helper = [&x]<std::size_t... Idx>(std::index_sequence<Idx...>) -> decltype(x) {
    return (decltype(x)(1) * ... * (static_cast<void>(Idx), x));
  };
  return f_helper(std::make_index_sequence<e>{});
}

/**
 * @brief Compute norm of each column in a matrix.
 *
 * Supports both sparse and dense matrices.
 */
auto colwise_norm(const auto & M)
  -> Eigen::Vector<typename std::decay_t<decltype(M)>::Scalar, std::decay_t<decltype(M)>::ColsAtCompileTime>
{
  using MType                     = std::decay_t<decltype(M)>;
  static constexpr bool is_sparse = std::is_base_of_v<Eigen::SparseMatrixBase<MType>, MType>;

  Eigen::Vector<typename std::decay_t<decltype(M)>::Scalar, std::decay_t<decltype(M)>::ColsAtCompileTime> ret;
  if constexpr (is_sparse) {
    ret.setZero(M.cols());
    for (auto i = 0; i < M.outerSize(); ++i) {
      for (Eigen::InnerIterator it(M, i); it; ++it) { ret(it.col()) += fpow<2>(it.value()); }
    }
    ret.noalias() = ret.cwiseSqrt();
  } else {
    ret = M.colwise().norm();
  }

  return ret;
}

SMOOTH_END_NAMESPACE
