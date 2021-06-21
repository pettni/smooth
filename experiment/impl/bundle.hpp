#ifndef BUNDLE_IMPL_HPP_
#define BUNDLE_IMPL_HPP_

#include <array>

#include <Eigen/Core>

#include "../../include/smooth/utils.hpp"
#include "common.hpp"

namespace smooth {

template<typename... Impl>
struct BundleImpl
{
  using Scalar = std::common_type_t<typename Impl::Scalar...>;

  static constexpr std::array<Eigen::Index, sizeof...(Impl)> RepSizes{Impl::RepSize...};
  static constexpr std::array<Eigen::Index, sizeof...(Impl)> Dofs{Impl::Dof...};
  static constexpr std::array<Eigen::Index, sizeof...(Impl)> Dims{Impl::Dim...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);
  static constexpr auto DimsPsum     = smooth::utils::array_psum(Dims);

  template<std::size_t Idx>
  using PartImpl = std::tuple_element_t<Idx, std::tuple<Impl...>>;

  static constexpr auto RepSize = RepSizesPsum.back();
  static constexpr auto Dof     = DofsPsum.back();
  static constexpr auto Dim     = DimsPsum.back();

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::setIdentity(
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void setRandom(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::setRandom(
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setZero();
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::matrix(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        m_out.template block<std::get<i>(Dims), std::get<i>(Dims)>(
          std::get<i>(DimsPsum), std::get<i>(DimsPsum)));
    });
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::composition(
        g_in1.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_in2.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::inverse(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::log(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        a_out.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum)));
    });
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::Ad(g_in.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)),
        A_out.template block<std::get<i>(Dofs), std::get<i>(Dofs)>(
          std::get<i>(DofsPsum), std::get<i>(DofsPsum)));
    });
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::exp(a_in.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum)),
        g_out.template segment<std::get<i>(RepSizes)>(std::get<i>(RepSizesPsum)));
    });
  }

  static void hat(TRefIn & a_in, MRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::hat(
        a_in.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum)),
        A_out.template block<std::get<i>(Dims), std::get<i>(Dims)>(std::get<i>(DimsPsum), std::get<i>(DimsPsum))
      );
    });
  }

  static void vee(MRefIn & A_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(Impl)>([&](auto i) {
      PartImpl<i>::vee(
        A_in.template block<std::get<i>(Dims), std::get<i>(Dims)>(std::get<i>(DimsPsum), std::get<i>(DimsPsum)),
        a_out.template segment<std::get<i>(Dofs)>(std::get<i>(DofsPsum))
      );
    });
  }

  static void ad(TRefIn, TMapRefOut A_out) { A_out.setZero(); }

  static void dr_exp(TRefIn, TMapRefOut A_out) { A_out.setZero(); }

  static void dr_expinv(TRefIn, TMapRefOut A_out) { A_out.setZero(); }
};

}  // namespace smooth

#endif  // BUNDLE_IMPL_HPP_
