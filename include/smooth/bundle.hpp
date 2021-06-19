#ifndef SMOOTH__BUNDLE_HPP_
#define SMOOTH__BUNDLE_HPP_

#include "common.hpp"
#include "concepts.hpp"
#include "macro.hpp"
#include "meta.hpp"
#include "storage.hpp"

namespace smooth {

namespace detail {

template<typename T>
struct lie_info;

template<LieGroup G>
struct lie_info<G>
{
  static constexpr Eigen::Index lie_size   = G::RepSize;
  static constexpr Eigen::Index lie_dof    = G::Dof;
  static constexpr Eigen::Index lie_dim    = G::Dim;
  static constexpr Eigen::Index lie_actdim = G::ActDim;

  template<typename NewScalar>
  using CastType = typename G::template CastType<NewScalar>;
};

template<StaticRnLike G>
struct lie_info<G>
{
  static constexpr Eigen::Index lie_size   = G::SizeAtCompileTime;
  static constexpr Eigen::Index lie_dof    = G::SizeAtCompileTime;
  static constexpr Eigen::Index lie_dim    = lie_size == -1 ? Eigen::Index(-1) : G::SizeAtCompileTime + 1;
  static constexpr Eigen::Index lie_actdim = G::SizeAtCompileTime;

  template<typename NewScalar>
  using CastType = Eigen::Matrix<NewScalar, G::SizeAtCompileTime, 1>;
};

}  // namespace detail

/**
 * @brief Bundle of multiple Lie types that can be treated as a single Lie group
 *
 * Bundle members can also be Eigen vectors by including Tn in the template argument list.
 */
template<MappableStorageLike _Storage, typename ... _Ts>
requires ((LieGroup<_Ts> || StaticRnLike<_Ts>) &&... && true)
   && (_Storage::Size == (detail::lie_info<_Ts>::lie_size + ...))
class BundleBase
{
private:

  _Storage s_;

  /* friend with other storage types */
  template<MappableStorageLike OS, typename... _OTs>
  requires ((LieGroup<_OTs> || StaticRnLike<_OTs>)&&... && true)
     && (OS::Size == (detail::lie_info<_OTs>::lie_size + ...))
  friend class BundleBase;

  static constexpr std::array<Eigen::Index, sizeof...(_Ts)>
    RepSizes{detail::lie_info<_Ts>::lie_size...},
    Dofs{detail::lie_info<_Ts>::lie_dof...},
    Dims{detail::lie_info<_Ts>::lie_dim...},
    ActDims{detail::lie_info<_Ts>::lie_actdim...};

  static constexpr auto RepSizesPsum = meta::array_psum(RepSizes);
  static constexpr auto DofsPsum     = meta::array_psum(Dofs);
  static constexpr auto DimsPsum     = meta::array_psum(Dims);
  static constexpr auto ActDimsPsum  = meta::array_psum(ActDims);

public:
  // REQUIRED CONSTANTS

  static constexpr auto RepSize = RepSizesPsum.back();
  static constexpr auto Dof     = DofsPsum.back();
  static constexpr auto Dim     = DimsPsum.back();
  static constexpr auto ActDim  = ActDimsPsum.back();

  // REQUIRED TYPES

  using Scalar = std::common_type_t<typename _Ts::Scalar ...>;
  using Storage = _Storage;

  static_assert(
    (std::is_same_v<Scalar, typename _Ts::Scalar> && ...),
    "Bundle members must have the same Scalar type"
  );

  using PlainObject = BundleBase<DefaultStorage<Scalar, RepSize>, _Ts...>;

  template<typename NewScalar>
  using CastType = BundleBase<
    DefaultStorage<Scalar, RepSize>,
    typename detail::lie_info<_Ts>::template CastType<NewScalar> ...
  >;

  template<MappableStorageLike NewStorage>
  using NewStorageType = BundleBase<NewStorage, _Ts ...>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  SMOOTH_COMMON_API(BundleBase)

  // BUNDLE-SPECIFIC API

  template<std::size_t Idx>
  using PartType = std::tuple_element_t<Idx, std::tuple<_Ts...>>;

  /**
   * @brief Construct from components
   */
  template<typename... S>
  requires ModifiableStorageLike<Storage>
    && (sizeof...(S) == sizeof...(_Ts))
    && std::conjunction_v<std::is_assignable<_Ts, S>...>
  explicit BundleBase(S &&... args)
  {
    meta::static_for<sizeof...(_Ts)>(
      [&](auto i) { part<i>() = std::get<i>(std::forward_as_tuple(args...)); }
    );
  }

  /**
   * @brief Access parts via map
   */
  template<std::size_t I>
  Map<PartType<I>> part() requires ModifiableStorageLike<Storage>
  {
    return Map<PartType<I>>(s_.data() + std::get<I>(RepSizesPsum));
  }

  /**
   * @brief Access parts via const map
   */
  template<std::size_t I>
  Map<const PartType<I>> part() const
  {
    return Map<const PartType<I>>(s_.data() + std::get<I>(RepSizesPsum));
  }

  // REQUIRED API

  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      if constexpr (StaticRnLike<PartType<i>>) {
        part<i>().setZero();
      } else {
        part<i>().setIdentity();
      }
    });
  }

  void setRandom() requires ModifiableStorageLike<Storage>
  {
    meta::static_for<sizeof...(_Ts)>([&](auto i) { part<i>().setRandom(); });
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    MatrixGroup ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dim_beg = std::get<i>(DimsPsum);
      static constexpr std::size_t dim_len = std::get<i>(Dims);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg).setIdentity();
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg)
          .template topRightCorner<PartType<i>::SizeAtCompileTime, 1>() = part<i>();
      } else {
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg) = part<i>().matrix_group();
      }
    });
    return ret;
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::SizeAtCompileTime == ActDim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    Vector ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t actdim_beg = std::get<i>(ActDimsPsum);
      static constexpr std::size_t actdim_len = std::get<i>(ActDims);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template segment<actdim_len>(actdim_beg) =
          part<i>() + x.template segment<actdim_len>(actdim_beg);
      } else {
        ret.template segment<actdim_len>(actdim_beg) =
          part<i>() * x.template segment<actdim_len>(actdim_beg);
      }
    });
    return ret;
  }

  /**
   * @brief Group composition
   */
  template<StorageLike OS>
  PlainObject operator*(const NewStorageType<OS> & r) const
  {
    PlainObject ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template part<i>() = part<i>() + r.template part<i>();
      } else {
        ret.template part<i>() = part<i>() * r.template part<i>();
      }
    });
    return ret;
  }

  /**
   * @brief Group inverse
   */
  PlainObject inverse() const
  {
    PlainObject ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template part<i>() = -part<i>();
      } else {
        ret.template part<i>() = part<i>().inverse();
      }
    });
    return ret;
  }

  /**
   * @brief Group logarithm
   */
  Tangent log() const
  {
    Tangent ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template segment<dof_len>(dof_beg) = part<i>();
      } else {
        ret.template segment<dof_len>(dof_beg) = part<i>().log();
      }
    });
    return ret;
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg).setIdentity();
      } else {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg) = part<i>().Ad();
      }
    });
    return ret;
  }

  // REQUIRED TANGENT API

  /**
   * @brief Group exponential
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    PlainObject ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template part<i>() = a.template segment<dof_len>(dof_beg);
      } else {
        ret.template part<i>() = PartType<i>::exp(a.template segment<dof_len>(dof_beg));
      }
    });
    return ret;
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        // ad is zero
      } else {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg) =
          PartType<i>::ad(a.template segment<dof_len>(dof_beg));
      }
    });
    return ret;
  }

  /**
   * @brief Algebra hat
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
  {
    MatrixGroup ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      static constexpr std::size_t dim_beg = std::get<i>(DimsPsum);
      static constexpr std::size_t dim_len = std::get<i>(Dims);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg)
          .template topRightCorner<PartType<i>::RowsAtCompileTime, 1>() =
          a.template segment<dof_len>(dof_beg);
      } else {
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg) =
          PartType<i>::hat(a.template segment<dof_len>(dof_beg));
      }
    });
    return ret;
  }

  /**
   * @brief Algebra vee
   */
  template<typename Derived>
  requires(Derived::RowsAtCompileTime == Dim && Derived::ColsAtCompileTime == Dim)
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    Tangent ret;
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      static constexpr std::size_t dim_beg = std::get<i>(DimsPsum);
      static constexpr std::size_t dim_len = std::get<i>(Dims);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template segment<dof_len>(dof_beg) =
          A.template block<dim_len, dim_len>(dim_beg, dim_beg)
            .template topRightCorner<PartType<i>::RowsAtCompileTime, 1>();
      } else {
        ret.template segment<dof_len>(dof_beg) =
          PartType<i>::vee(A.template block<dim_len, dim_len>(dim_beg, dim_beg));
      }
    });
    return ret;
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg).setIdentity();
      } else {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg) =
          PartType<i>::dr_exp(a.template segment<dof_len>(dof_beg));
      }
    });
    return ret;
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == Dof)
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Ts)>([&](auto i) {
      static constexpr std::size_t dof_beg = std::get<i>(DofsPsum);
      static constexpr std::size_t dof_len = std::get<i>(Dofs);
      if constexpr (StaticRnLike<PartType<i>>) {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg).setIdentity();
      } else {
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg) =
          PartType<i>::dr_expinv(a.template segment<dof_len>(dof_beg));
      }
    });
    return ret;
  }
};

// Typedef for bundle with regular storage
template<typename ... _Ts>
requires((LieGroup<_Ts> || StaticRnLike<_Ts>) &&... && true)
using Bundle = BundleBase<
  DefaultStorage<std::common_type_t<typename _Ts::Scalar ...>, (detail::lie_info<_Ts>::lie_size + ...)>,
  _Ts...
>;
}  // namespace smooth

#endif  // SMOOTH__BUNDLE_HPP_
