#ifndef BUNDLE_IMPL_HPP_
#define BUNDLE_IMPL_HPP_

#include <array>

#include <Eigen/Core>

#include "../../include/smooth/utils.hpp"

namespace smooth {

template<typename... Impl>
struct BundleImpl
{
  using Scalar = std::common_type_t<typename Impl::Scalar...>;

  static constexpr std::array<Eigen::Index, sizeof...(Impl)> RepSizes{Impl::RepSize...};
  static constexpr std::array<Eigen::Index, sizeof...(Impl)> Dofs{Impl::Dof...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);

  // REQUIRED CONSTANTS

  template<std::size_t Idx>
  using PartImpl = std::tuple_element_t<Idx, std::tuple<Impl...>>;

  static constexpr auto RepSize = RepSizesPsum.back();
  static constexpr auto Dof     = DofsPsum.back();

  static void setIdentity(Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::setIdentity(
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void setRandom(Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::setRandom(
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  template<typename Derived1, typename Derived2>
  static void composition(const Eigen::MatrixBase<Derived1> & g_in1,
    const Eigen::MatrixBase<Derived2> & g_in2,
    Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::composition(
        g_in1.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_in2.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  template<typename Derived>
  static void inverse(
    const Eigen::MatrixBase<Derived> & g_in, Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::inverse(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  template<typename Derived>
  static void log(
    const Eigen::MatrixBase<Derived> & g_in, Eigen::Ref<Eigen::Matrix<Scalar, Dof, 1>> a_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::log(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        a_out.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum)));
    });
  }

  template<typename Derived>
  static void Ad(
    const Eigen::MatrixBase<Derived> & g_in, Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof>> A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::Ad(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        A_out.template block<std::get<i>(Dofs), std::get<i>(Dofs)>(
          std::get<i>(DofsPsum), std::get<i>(DofsPsum)));
    });
  }

  template<typename Derived>
  static void exp(
    const Eigen::MatrixBase<Derived> & a_in, Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>> g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::exp(a_in.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }
};

}  // namespace smooth

#endif  // BUNDLE_IMPL_HPP_
