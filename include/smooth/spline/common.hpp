// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <optional>

#include <Eigen/Core>

#include "../concepts/lie_group.hpp"

namespace smooth {
inline namespace v1_0 {

/// @brief Optional argument for spline time derivatives
template<LieGroup G>
using OptTangent = std::optional<Eigen::Ref<Tangent<G>>>;

/// @brief Jacobian of order K spline value w.r.t. coefficients.
template<LieGroup G, int K>
using SplineJacobian = Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> == -1 ? -1 : Dof<G> *(K + 1)>;

/// @brief Optional argument for Jacobian of spline w.r.t. coefficients.
template<LieGroup G, int K>
using OptSplineJacobian = std::optional<Eigen::Ref<SplineJacobian<G, K>>>;

}  // namespace v1_0
}  // namespace smooth
