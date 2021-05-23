#ifndef SMOOTH__BUNDLE_HPP_
#define SMOOTH__BUNDLE_HPP_

#include "concepts.hpp"
#include "common.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"
#include "meta.hpp"
#include "storage.hpp"


namespace smooth
{

/**
 * @brief Bundle of multiple Lie types that can be treated as a single Lie group
 *
 * Bundle members can also be Eigen vectors by including En in the template argument list.
 */
template<typename _Scalar, MappableStorageLike _Storage, template<typename> typename ... _Gs>
requires(
  ((LieGroupLike<_Gs<_Scalar>>|| RnLike<_Gs<_Scalar>>) && ... && true) &&
  (_Storage::SizeAtCompileTime ==
  meta::iseq_sum_v<std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_size ...>>) &&
  std::is_same_v<typename _Storage::Scalar, _Scalar>
)
class BundleBase
  : public LieGroupBase<
    BundleBase<_Scalar, _Storage, _Gs...>,
    meta::iseq_sum_v<std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_size ...>>
  >
{
private:
  _Storage s_;

  using lie_sizes = std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_size ...>;
  using lie_dofs = std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_dof ...>;
  using lie_dims = std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_dim ...>;
  using lie_actdims = std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_actdim ...>;

  using lie_sizes_psum = meta::iseq_psum_t<lie_sizes>;
  using lie_dofs_psum = meta::iseq_psum_t<lie_dofs>;
  using lie_dims_psum = meta::iseq_psum_t<lie_dims>;
  using lie_actdims_psum = meta::iseq_psum_t<lie_actdims>;

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t lie_size = meta::iseq_sum_v<lie_sizes>;
  static constexpr uint32_t lie_dof = meta::iseq_sum_v<lie_dofs>;
  static constexpr uint32_t lie_dim = meta::iseq_sum_v<lie_dims>;
  static constexpr uint32_t lie_actdim = meta::iseq_sum_v<lie_actdims>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  SMOOTH_BUNDLE_BOILERPLATE(BundleBase)

  // BUNDLE-SPECIFIC API

  template<std::size_t Idx>
  using PartType = std::tuple_element_t<Idx, std::tuple<_Gs<_Scalar>...>>;

  /**
   * @brief Construct from components
   */
  template<typename ... S>
  explicit BundleBase(S && ... args)
  requires ModifiableStorageLike<Storage>&&
  (sizeof...(S) == sizeof...(_Gs)) &&
  std::conjunction_v<std::is_assignable<_Gs<_Scalar>, S>...>
  {
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        part<i>() = std::get<i>(std::forward_as_tuple(args ...));
      });
  }

  /**
   * @brief Access parts via map
   */
  template<std::size_t I>
  Map<PartType<I>> part()
  requires ModifiableStorageLike<Storage>
  {
    return Map<PartType<I>>(s_.data() + meta::iseq_el_v<I, lie_sizes_psum>);
  }

  /**
   * @brief Access parts via const map
   */
  template<std::size_t I>
  Map<const PartType<I>> part() const
  {
    return Map<const PartType<I>>(s_.data() + meta::iseq_el_v<I, lie_sizes_psum>);
  }

  // REQUIRED API

  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (RnLike<PartType<i>>) {
          part<i>().setZero();
        } else {
          part<i>().setIdentity();
        }
      });
  }

  void setRandom() requires ModifiableStorageLike<Storage>
  {
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (RnLike<PartType<i>>) {
          part<i>().setRandom();
        } else {
          part<i>().setRandom();
        }
      });
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    MatrixGroup ret;
    ret.setZero();
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dim_beg = meta::iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = meta::iseq_el_v<i, lie_dims>;
        if constexpr (RnLike<PartType<i>>) {
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
  requires(Derived::SizeAtCompileTime == lie_actdim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    Vector ret;
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t actdim_beg = meta::iseq_el_v<i, lie_actdims_psum>;
        static constexpr std::size_t actdim_len = meta::iseq_el_v<i, lie_actdims>;
        if constexpr (RnLike<PartType<i>>) {
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
  template<typename OS, template<typename> typename ... _OGs>
  Group operator*(const BundleBase<Scalar, OS, _OGs...> & r) const
  {
    Group ret;
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (RnLike<PartType<i>>) {
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
  Group inverse() const
  {
    Group ret;
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (RnLike<PartType<i>>) {
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
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
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
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
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
  static Group exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    Group ret;
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
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
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
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
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    MatrixGroup ret;
    ret.setZero();
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        static constexpr std::size_t dim_beg = meta::iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = meta::iseq_el_v<i, lie_dims>;
        if constexpr (RnLike<PartType<i>>) {
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
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  requires(Derived::RowsAtCompileTime == lie_dim && Derived::ColsAtCompileTime == lie_dim)
  {
    Tangent ret;
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        static constexpr std::size_t dim_beg = meta::iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = meta::iseq_el_v<i, lie_dims>;
        if constexpr (RnLike<PartType<i>>) {
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
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
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
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    TangentMap ret;
    ret.setZero();
    meta::static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = meta::iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = meta::iseq_el_v<i, lie_dofs>;
        if constexpr (RnLike<PartType<i>>) {
          ret.template block<dof_len, dof_len>(dof_beg, dof_beg).setIdentity();
        } else {
          ret.template block<dof_len, dof_len>(dof_beg, dof_beg) =
          PartType<i>::dr_expinv(a.template segment<dof_len>(dof_beg));
        }
      });
    return ret;
  }
};

template<typename _Scalar, template<typename> typename ... _Gs>
using Bundle = BundleBase<
  _Scalar,
  DefaultStorage<_Scalar,
  meta::iseq_sum<std::index_sequence<detail::lie_info<_Gs<_Scalar>>::lie_size ...>>::value>,
  _Gs...
>;

}  // namespace smooth

#endif  // SMOOTH__BUNDLE_HPP_
