#ifndef SMOOTH__COMPAT__CERES_HPP_
#define SMOOTH__COMPAT__CERES_HPP_

#include <ceres/autodiff_local_parameterization.h>

#include "smooth/concepts.hpp"
#include "smooth/storage.hpp"


namespace smooth
{

template<LieGroup G>
struct ParameterizationFunctor
{
  template<typename Scalar>
  bool operator()(const Scalar * x, const Scalar * delta, Scalar * x_plus_delta) const
  {
    using GScalar = decltype(G().template cast<Scalar>());

    smooth::Map<const GScalar> mx(x);
    Eigen::Map<const typename GScalar::Tangent> mdelta(delta);
    smooth::Map<GScalar> mx_plus_delta(x_plus_delta);

    mx_plus_delta = mx + mdelta;
    return true;
  }
};

template<LieGroup G>
class LieGroupParameterization
  : public ceres::AutoDiffLocalParameterization<ParameterizationFunctor<G>, G::RepSize, G::Dof>
{};

}  // namespace smooth

#endif  // SMOOTH__COMPAT__CERES_HPP_
