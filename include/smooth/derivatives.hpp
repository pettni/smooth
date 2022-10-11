// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "lie_groups.hpp"

/**
 * @file
 * @brief Various useful derivatives.
 */

namespace smooth {

/**
 * @brief Derivative of matrix product.
 *
 * @param A matrix                                         [N x K]
 * @param dA derivative of A on horizontal Hessian form    [K x N*Nvar]
 * @param B matrix                                         [K x M]
 * @param dB derivative of B on horizontal Hessian form    [M x K*Nvar]
 * @return derivative of A * B on horizontal Hessian form
 */
template<typename At, typename dAt, typename Bt, typename dBt>
auto d_matrix_product(const At & A, const dAt & dA, const Bt & B, const dBt & dB);

/**
 * @brief Hessian of composed function \f$ (f \circ g)(x) \f$.
 *
 * @param Jf Jacobian of f at y = g(x)  [No x Ny   ]
 * @param Hf Hessian of f at y = g(x)   [Ny x No*Ny]
 * @param Jg Jacobian of g at x         [Ny x Nx   ]
 * @param Hg Hessian of g at x          [Nx x Ny*Nx]
 *
 * @return Hessian of size [No x No*Nx]
 */
template<typename JfT, typename HfT, typename JgT, typename HgT>
auto d2_fog(const JfT & Jf, const HfT & Hf, const JgT & Jg, const HgT & Hg);

/**
 * @brief Jacobian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{r} (x \ominus_r y)_{x} \f$
 */
template<LieGroup G>
TangentMap<G> dr_rminus(const Tangent<G> & e);

/**
 * @brief Hessian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} (x \ominus_r y)_{xx} \f$
 */
template<LieGroup G>
Hessian<G> d2r_rminus(const Tangent<G> & e);

/**
 * @brief Jacobian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^r \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_x \f$
 */
template<LieGroup G>
Eigen::RowVector<Scalar<G>, Dof<G>> dr_rminus_squarednorm(const Tangent<G> & e);

/**
 * @brief Hessian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_{xx} \f$
 */
template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>> d2r_rminus_squarednorm(const Tangent<G> & e);

}  // namespace smooth

#include "detail/derivatives_impl.hpp"
