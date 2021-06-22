#ifndef SMOOTH__IMPL__BUNDLE_HPP_
#define SMOOTH__IMPL__BUNDLE_HPP_

#include <array>

#include <Eigen/Core>

#include "../utils.hpp"
#include "common.hpp"

using std::get;

/**
 * @brief Bundle Lie group
 *
 * Represents the direct product
 *  G1 x G2 x ... x Gk
 * for Lie groups G1 ... Gk.
 */
namespace smooth {

template<typename... GsImpl>
struct BundleImpl
{
  using Scalar = std::common_type_t<typename GsImpl::Scalar...>;

  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> RepSizes{GsImpl::RepSize...};
  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> Dofs{GsImpl::Dof...};
  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> Dims{GsImpl::Dim...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);
  static constexpr auto DimsPsum     = smooth::utils::array_psum(Dims);

  template<std::size_t Idx>
  using PartImpl = std::tuple_element_t<Idx, std::tuple<GsImpl...>>;

  static constexpr auto RepSize = RepSizesPsum.back();
  static constexpr auto Dof     = DofsPsum.back();
  static constexpr auto Dim     = DimsPsum.back();

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setIdentity(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)));
    });
  }

  static void setRandom(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setRandom(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)));
    });
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::matrix(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        m_out.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum))
      );
    });
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::composition(
        g_in1.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_in2.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );
    });
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::inverse(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );
    });
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::log(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        a_out.template segment<get<i>(Dofs)>(get<i>(DofsPsum))
      );
    });
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::Ad(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
      );
    });
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::exp(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );
    });
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::hat(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        A_out.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum))
      );
    });
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::vee(
        A_in.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum)),
        a_out.template segment<get<i>(Dofs)>(get<i>(DofsPsum))
      );
    });
  }

  static void ad(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::ad(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
      );
    });
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::dr_exp(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
      );
    });
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::dr_expinv(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
      );
    });
  }
};

}  // namespace smooth

#endif  // SMOOTH__IMPL__BUNDLE_HPP_
