// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__INTERNAL__BUNDLE_HPP_
#define SMOOTH__INTERNAL__BUNDLE_HPP_

#include <array>

#include <Eigen/Core>

#include "common.hpp"
#include "smooth/internal/utils.hpp"

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

  static_assert(
    (std::is_same_v<Scalar, typename GsImpl::Scalar> && ...),
    "Implementation Scalar types must be the same");

  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> RepSizes{GsImpl::RepSize...};
  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> Dofs{GsImpl::Dof...};
  static constexpr std::array<Eigen::Index, sizeof...(GsImpl)> Dims{GsImpl::Dim...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);
  static constexpr auto DimsPsum     = smooth::utils::array_psum(Dims);

  template<std::size_t Idx>
  using PartImpl = std::tuple_element_t<Idx, std::tuple<GsImpl...>>;

  static constexpr Eigen::Index RepSize = RepSizesPsum.back();
  static constexpr Eigen::Index Dof     = DofsPsum.back();
  static constexpr Eigen::Index Dim     = DimsPsum.back();

  SMOOTH_DEFINE_REFS;

  // clang-format off

  static void setIdentity(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setIdentity(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );
    });
  }

  static void setRandom(GRefOut g_out)
  {
    smooth::utils::static_for<sizeof...(GsImpl)>([&](auto i) {
      PartImpl<i>::setRandom(
        g_out.template segment<get<i>(RepSizes)>(get<i>(RepSizesPsum))
      );
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

  // clang-format on
};

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__BUNDLE_HPP_
