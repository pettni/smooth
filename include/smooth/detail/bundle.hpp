// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <array>
#include <tuple>

#include <Eigen/Core>

#include "common.hpp"
#include "utils.hpp"

/**
 * @brief Bundle Lie group
 *
 * Represents the direct product
 *  G1 x G2 x ... x Gk
 * for Lie groups G1 ... Gk.
 */
SMOOTH_BEGIN_NAMESPACE

using std::get;

template<typename... GsImpl>
struct BundleImpl
{
  using Scalar = std::common_type_t<typename GsImpl::Scalar...>;

  static_assert(
    (std::is_same_v<Scalar, typename GsImpl::Scalar> && ...), "Implementation Scalar types must be the same");

  static constexpr std::array<int, sizeof...(GsImpl)> RepSizes{GsImpl::RepSize...};
  static constexpr std::array<int, sizeof...(GsImpl)> Dofs{GsImpl::Dof...};
  static constexpr std::array<int, sizeof...(GsImpl)> Dims{GsImpl::Dim...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);
  static constexpr auto DimsPsum     = smooth::utils::array_psum(Dims);

  template<std::size_t Idx>
  using PartImpl = std::tuple_element_t<Idx, std::tuple<GsImpl...>>;

  static constexpr auto BundleSize = sizeof...(GsImpl);

  static constexpr auto RepSize       = RepSizesPsum.back();
  static constexpr auto Dof           = DofsPsum.back();
  static constexpr auto Dim           = DimsPsum.back();
  static constexpr bool IsCommutative = (GsImpl::IsCommutative && ...);

  SMOOTH_DEFINE_REFS;

  // clang-format off

  static void setIdentity(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setIdentity(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );  //NOLINT
    });
  }

  static void setRandom(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setRandom(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      ); //NOLINT
    });
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::matrix(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        m_out.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum))
      ); //NOLINT
    });
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::composition(
        g_in1.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_in2.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      ); //NOLINT
    });
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::inverse(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      ); //NOLINT
    });
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::log(
        g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
        a_out.template segment<get<i>(Dofs)>(get<i>(DofsPsum))
      ); //NOLINT
    });
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        PartImpl<i>::Ad(
          g_in.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum)),
          A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
        ); //NOLINT
      } else {
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum)).setIdentity();
      }
    });
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::exp(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      ); //NOLINT
    });
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::hat(
        a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
        A_out.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum))
      ); //NOLINT
    });
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::vee(
        A_in.template block<get<i>(Dims), get<i>(Dims)>(get<i>(DimsPsum), get<i>(DimsPsum)),
        a_out.template segment<get<i>(Dofs)>(get<i>(DofsPsum))
      ); //NOLINT
    });
  }

  static void ad(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        PartImpl<i>::ad(
          a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
          A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
        ); //NOLINT
      }
    });
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        PartImpl<i>::dr_exp(
          a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
          A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
        ); //NOLINT
      } else {
          A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum)).setIdentity();
      }
    });
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        PartImpl<i>::dr_expinv(
          a_in.template segment<get<i>(Dofs)>(get<i>(DofsPsum)),
          A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum))
        ); //NOLINT
      } else {
        A_out.template block<get<i>(Dofs), get<i>(Dofs)>(get<i>(DofsPsum), get<i>(DofsPsum)).setIdentity();
      }
    });
  }

  static void d2r_exp(TRefIn a_in, THessRefOut H_out) {
    H_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        static constexpr auto Bi = get<i>(DofsPsum);  // block start
        static constexpr auto Di = get<i>(Dofs);  // block size
        Eigen::Matrix<Scalar, Di, Di * Di> Hi;
        PartImpl<i>::d2r_exp(a_in.template segment<Di>(Bi), Hi);
        for (auto j = 0u; j < Di; ++j) {
          H_out.template block<Di, Di>(Bi, Dof * (Bi + j) + Bi) = Hi.template middleCols<Di>(Di * j);
        }
      }
    });
  }

  static void d2r_expinv(TRefIn a_in, THessRefOut H_out) {
    H_out.setZero();
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      if constexpr (!PartImpl<i>::IsCommutative) {
        static constexpr auto Bi = get<i>(DofsPsum);  // block start
        static constexpr auto Di = get<i>(Dofs);  // block size
        Eigen::Matrix<Scalar, Di, Di * Di> Hi;
        PartImpl<i>::d2r_expinv(a_in.template segment<Di>(Bi), Hi);
        for (auto j = 0u; j < Di; ++j) {
          H_out.template block<Di, Di>(Bi, Dof * (Bi + j) + Bi) = Hi.template middleCols<Di>(Di * j);
        }
      }
    });
  }

  // clang-format on
};

SMOOTH_END_NAMESPACE
