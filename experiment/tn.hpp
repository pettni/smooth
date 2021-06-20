#include "lie_group.hpp"

template<Eigen::Index N>
class TnTag {};

template<Eigen::Index N>
struct lie_impl<TnTag<N>>
{
  static constexpr Eigen::Index Dof     = N;
  static constexpr Eigen::Index RepSize = N;

  template<typename Derived>
  static void log(
      const Eigen::ArrayBase<Derived> & s,
      Eigen::Ref<Eigen::Matrix<typename Derived::Scalar, N, 1>> a
  )
  {
    a = s;
  }

  template<typename Derived>
  static void exp(
    const Eigen::MatrixBase<Derived> & a,
    Eigen::Ref<Eigen::Array<typename Derived::Scalar, N, 1>> s
  )
  {
    s = a;
  }
};

template<Eigen::Index N, typename Scalar>
using T = LieGroup<Scalar, TnTag<N>>;

