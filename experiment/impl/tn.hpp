#ifndef IMPL__TN_HPP_
#define IMPL__TN_HPP_

#include <array>

#include <Eigen/Core>

namespace smooth {

template<int N, typename _Scalar>
struct TnImpl
{
  using Scalar = _Scalar;

  static constexpr Eigen::Index Dof     = N;
  static constexpr Eigen::Index RepSize = N;

  static void setIdentity(Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out) { g_out.setZero(); }

  static void setRandom(Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out) { g_out.setRandom(); }

  template<typename Derived1, typename Derived2>
  static void composition(const Eigen::MatrixBase<Derived1> & g_in1,
    const Eigen::MatrixBase<Derived2> & g_in2,
    Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    g_out = g_in1 + g_in2;
  }

  template<typename Derived>
  static void inverse(
    const Eigen::MatrixBase<Derived> & g_in, Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    g_out = -g_in;
  }

  template<typename Derived>
  static void log(
    const Eigen::MatrixBase<Derived> & g_in, Eigen::Ref<Eigen::Matrix<Scalar, Dof, 1>> a_out)
  {
    a_out = g_in;
  }

  template<typename Derived>
  static void Ad(
    const Eigen::MatrixBase<Derived> &, Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof>> A_out)
  {
    A_out.setZero();
  }

  template<typename Derived>
  static void exp(
    const Eigen::MatrixBase<Derived> & a_in, Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    g_out = a_in;
  }
};

}  // namespace smooth

#endif  // IMPL__TN_HPP_
