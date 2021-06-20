#include "../include/smooth/utils.hpp"
#include "lie_group.hpp"

template<typename... Tags>
class BundleTag {};

template<typename... Tags>
struct lie_impl<BundleTag<Tags...>>
{
  static constexpr std::array<Eigen::Index, sizeof...(Tags)>
    RepSizes{lie_impl<Tags>::RepSize...},
    Dofs{lie_impl<Tags>::Dof...};

  static constexpr auto RepSizesPsum = smooth::utils::array_psum(RepSizes);
  static constexpr auto DofsPsum     = smooth::utils::array_psum(Dofs);

  // REQUIRED CONSTANTS

  template<std::size_t Idx>
  using PartTag = std::tuple_element_t<Idx, std::tuple<Tags...>>;

  static constexpr auto RepSize = RepSizesPsum.back();
  static constexpr auto Dof     = DofsPsum.back();

  template<typename Derived>
  static void exp(
    const Eigen::MatrixBase<Derived> & a,
    Eigen::Ref<Eigen::Array<typename Derived::Scalar, RepSize, 1>> c
  )
  {
    smooth::utils::static_for<sizeof...(Tags)>([&] (auto i) {
        static constexpr std::size_t dofbeg = std::get<i>(DofsPsum);
        static constexpr std::size_t doflen = std::get<i>(Dofs);
        static constexpr std::size_t cofbeg = std::get<i>(RepSizesPsum);
        static constexpr std::size_t coflen= std::get<i>(RepSizes);
        lie_impl<PartTag<i>>::exp(
            a.template segment<doflen>(dofbeg),
            c.template segment<coflen>(cofbeg)
        );
    });
  }

  template<typename Derived>
  static void log(
    const Eigen::ArrayBase<Derived> & c,
    Eigen::Ref<Eigen::Matrix<typename Derived::Scalar, Dof, 1>> a)
  {
    smooth::utils::static_for<sizeof...(Tags)>([&] (auto i) {
        static constexpr std::size_t dofbeg = std::get<i>(DofsPsum);
        static constexpr std::size_t doflen = std::get<i>(Dofs);
        static constexpr std::size_t cofbeg = std::get<i>(RepSizesPsum);
        static constexpr std::size_t coflen= std::get<i>(RepSizes);
        lie_impl<PartTag<i>>::log(
            c.template segment<coflen>(cofbeg),
            a.template segment<doflen>(dofbeg)
        );
    });
  }
};

template<typename Scalar, typename... Tags>
class Bundle : public LieGroup<Scalar, BundleTag<Tags...>>
{
  using Base = LieGroup<Scalar, BundleTag<Tags...>>;
};

