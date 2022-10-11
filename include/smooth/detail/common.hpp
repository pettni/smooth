// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

namespace smooth {

static constexpr double eps2 = 1e-8;

#define SMOOTH_DEFINE_REFS                                                      \
  using GRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, RepSize, 1>> &;  \
  using GRefOut = Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>>;                \
                                                                                \
  using TRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, 1>> &;      \
  using TRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, 1>>;                    \
                                                                                \
  using TMapRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, Dof>> &; \
  using TMapRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof>>;               \
                                                                                \
  using THessRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof * Dof>>;        \
                                                                                \
  using MRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dim, Dim>> &;    \
  using MRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>;                  \
                                                                                \
  static_assert(true)

}  // namespace smooth
