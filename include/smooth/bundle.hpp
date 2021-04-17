#ifndef SMOOTH__BUNDLE_HPP_
#define SMOOTH__BUNDLE_HPP_

#include "concepts.hpp"
#include "common.hpp"
#include "storage.hpp"


namespace smooth
{

template<typename T>
struct lie_info;

template<LieGroupLike G>
struct lie_info<G>
{
  static constexpr uint32_t lie_size = G::lie_size;
  static constexpr uint32_t lie_dof = G::lie_dof;
  static constexpr uint32_t lie_dim = G::lie_dim;
  static constexpr uint32_t lie_actdim = G::lie_actdim;
};

template<EnLike G>
struct lie_info<G>
{
  static constexpr uint32_t lie_size = G::SizeAtCompileTime;
  static constexpr uint32_t lie_dof = G::SizeAtCompileTime;
  static constexpr uint32_t lie_dim = G::SizeAtCompileTime + 1;
  static constexpr uint32_t lie_actdim = G::SizeAtCompileTime;
};


template<typename _Scalar, MappableStorageLike _Storage, template<typename> typename ... _Gs>
requires(
  ((LieGroupLike<_Gs<_Scalar>>|| EnLike<_Gs<_Scalar>>) && ... && true) &&
  (_Storage::SizeAtCompileTime ==
  iseq_sum<std::index_sequence<lie_info<_Gs<_Scalar>>::lie_size ...>>::value) &&
  std::is_same_v<typename _Storage::Scalar, _Scalar>
)
class BundleBase
{
private:
  _Storage s_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename OtherScalar, MappableStorageLike OS, template<typename> typename ... Gs>
  requires std::is_same_v<_Scalar, OtherScalar>
  friend class BundleBase;

  using lie_sizes = std::index_sequence<lie_info<_Gs<_Scalar>>::lie_size ...>;
  using lie_dofs = std::index_sequence<lie_info<_Gs<_Scalar>>::lie_dof ...>;
  using lie_dims = std::index_sequence<lie_info<_Gs<_Scalar>>::lie_dim ...>;
  using lie_actdims = std::index_sequence<lie_info<_Gs<_Scalar>>::lie_actdim ...>;

  using lie_sizes_psum = typename iseq_psum<lie_sizes>::type;
  using lie_dofs_psum = typename iseq_psum<lie_dofs>::type;
  using lie_dims_psum = typename iseq_psum<lie_dims>::type;
  using lie_actdims_psum = typename iseq_psum<lie_actdims>::type;

public:
  static constexpr uint32_t lie_size = iseq_sum<lie_sizes>::value;
  static constexpr uint32_t lie_dof = iseq_sum<lie_dofs>::value;
  static constexpr uint32_t lie_dim = iseq_sum<lie_dims>::value;
  static constexpr uint32_t lie_actdim = iseq_sum<lie_actdims>::value;

  using Scalar = _Scalar;
  using Storage = _Storage;

  using Group = BundleBase<Scalar, DefaultStorage<Scalar, lie_size>, _Gs...>;
  using Tangent = Eigen::Matrix<Scalar, lie_dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, lie_dof, lie_dof>;
  using Vector = Eigen::Matrix<Scalar, lie_actdim, 1>;
  using MatrixGroup = Eigen::Matrix<Scalar, lie_dim, lie_dim>;

  // BUNDLE-SPECIFIC TYPES

  template<std::size_t Idx>
  using PartType = std::tuple_element_t<Idx, std::tuple<_Gs<_Scalar>...>>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  BundleBase() = default;
  BundleBase(const BundleBase & o) = default;
  BundleBase(BundleBase && o) = default;
  BundleBase & operator=(const BundleBase & o) = default;
  BundleBase & operator=(BundleBase && o) = default;
  ~BundleBase() = default;

  /**
   * @brief Copy constructor from other storage types
   */
  template<StorageLike OS>
  BundleBase(const BundleBase<Scalar, OS, _Gs...> & o)
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});
  }

  /**
   * @brief Forwarding constructor to storage for map types
   */
  explicit BundleBase(Scalar * ptr)
  requires std::is_constructible_v<Storage, Scalar *>
  : s_(ptr) {}

  /**
   * @brief Forwarding constructor to storage for const map types
   */
  explicit BundleBase(const Scalar * ptr)
  requires std::is_constructible_v<Storage, const Scalar *>
  : s_(ptr) {}

  /**
   * @brief Copy assignment from other BundleBase
   */
  template<StorageLike OS>
  BundleBase & operator=(const BundleBase<Scalar, OS, _Gs...> & o)
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  // BUNDLE-SPECIFIC API

  /**
   * @brief Construct from components
   */
  template<typename ... S>
  explicit BundleBase(S && ... args)
  requires (sizeof...(S) == sizeof...(_Gs))
  && std::conjunction_v<std::is_assignable<_Gs<_Scalar>, S> ...>
  {
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        part<i>() = std::get<i>(std::forward_as_tuple(args...));
      });
  }

  /**
   * @brief Access parts via map
   */
  template<std::size_t I>
  Map<PartType<I>> part()
  requires ModifiableStorageLike<Storage>
  {
    return Map<PartType<I>>(s_.data() + iseq_el_v<I, lie_sizes_psum>);
  }

  /**
   * @brief Access parts via const map
   */
  template<std::size_t I>
  ConstMap<PartType<I>> part() const
  {
    return ConstMap<PartType<I>>(s_.data() + iseq_el_v<I, lie_sizes_psum>);
  }

  // LIE GROUP BASE API

  static Group Identity()
  {
    Group ret;
    ret.setIdentity();
    return ret;
  }

  template<typename RNG>
  static Group Random(RNG & rng)
  {
    Group ret;
    ret.setRandom(rng);
    return ret;
  }

  template<typename NewScalar>
  BundleBase<NewScalar, DefaultStorage<NewScalar, lie_size>, _Gs...> cast() const
  {
    BundleBase<NewScalar, DefaultStorage<NewScalar, lie_size>, _Gs...> ret;
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        ret.template part<i>() = part<i>().template cast<NewScalar>();
      });
    return ret;
  }

  /**
   * @brief Compare two Lie group elements
   */
  template<typename Other>
  bool isApprox(
    const Other & o,
    const Scalar & eps = Eigen::NumTraits<Scalar>::dummy_precision()) const
  {
    bool ret = true;
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if (ret) {
          ret &= part<i>().isApprox(o.template part<i>(), eps);
        }
      });
    return ret;
  }

  /**
   * @brief Access group storage
   */
  Storage & coeffs()
  {
    return s_;
  }

  /**
   * @brief Const access group storage
   */
  const Storage & coeffs() const
  {
    return s_;
  }

  /**
   * @brief Access raw const data pointer
   */
  const Scalar * data() const requires MappableStorageLike<Storage>
  {
    return coeffs().data();
  }

  /**
   * @brief Access raw data pointer
   */
  Scalar * data() requires ModifiableStorageLike<Storage>
  {
    return coeffs().data();
  }

  /**
   * @brief Overload operator*= for inplace composition
   */
  template<StorageLike OS>
  BundleBase & operator*=(const BundleBase<Scalar, OS, _Gs...> & o)
  {
    *this = *this * o;
    return *this;
  }

  /**
   * @brief Overload operator+ for right-plus
   *
   * g + a := g1 * exp(a)
   */
  template<typename TangentDerived>
  auto operator+(const Eigen::MatrixBase<TangentDerived> & t) const
  {
    return *this * exp(t);
  }

  /**
   * @brief Overload operator+= for inplace right-plus
   *
   * g + a := g1 * exp(a)
   */
  template<typename TangentDerived>
  BundleBase & operator+=(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return *this *= exp(t);
  }

  /**
   * @brief Overload operator- for right-minus
   *
   * g1 - g2 := (g2.inverse() * g1).log()
   */
  template<StorageLike OS>
  auto operator-(const BundleBase<Scalar, OS, _Gs...> & o) const
  {
    return (o.inverse() * *this).log();
  }

  /**
   * @brief Left jacobian of the exponential
   */
  template<typename TangentDerived>
  static auto dl_exp(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (exp(t).Ad() * dr_exp(t)).eval();
  }

  /**
   * @brief Inverse of left jacobian of the exponential
   */
  template<typename TangentDerived>
  static auto dl_expinv(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (-ad(t) + dr_expinv(t)).eval();
  }

  // REQUIRED API

  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (EnLike<PartType<i>>) {
          part<i>().setZero();
        } else {
          part<i>().setIdentity();
        }
      });
  }

  template<typename RNG>
  void setRandom(RNG & rng) requires ModifiableStorageLike<Storage>
  {
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (EnLike<PartType<i>>) {
          part<i>() = PartType<i>::NullaryExpr([&rng](int) {return u_distr<Scalar>(rng);});
        } else {
          part<i>().setRandom(rng);
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dim_beg = iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = iseq_el_v<i, lie_dims>;
        if constexpr (EnLike<PartType<i>>) {
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t actdim_beg = iseq_el_v<i, lie_actdims_psum>;
        static constexpr std::size_t actdim_len = iseq_el_v<i, lie_actdims>;
        if constexpr (EnLike<PartType<i>>) {
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (EnLike<PartType<i>>) {
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        if constexpr (EnLike<PartType<i>>) {
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
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
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename TangentDerived>
  static Group exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    Group ret;
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    ret.setZero();
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename TangentDerived>
  static MatrixGroup hat(const Eigen::MatrixBase<TangentDerived> & a)
  {
    MatrixGroup ret;
    ret.setZero();
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        static constexpr std::size_t dim_beg = iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = iseq_el_v<i, lie_dims>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename AlgebraDerived>
  static Tangent vee(const Eigen::MatrixBase<AlgebraDerived> & A)
  {
    Tangent ret;
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        static constexpr std::size_t dim_beg = iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = iseq_el_v<i, lie_dims>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    ret.setZero();
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
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
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    TangentMap ret;
    ret.setZero();
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t dof_beg = iseq_el_v<i, lie_dofs_psum>;
        static constexpr std::size_t dof_len = iseq_el_v<i, lie_dofs>;
        if constexpr (EnLike<PartType<i>>) {
          ret.template block<dof_len, dof_len>(dof_beg, dof_beg).setIdentity();
        } else {
          ret.template block<dof_len, dof_len>(dof_beg, dof_beg) =
          PartType<i>::dr_expinv(a.template segment<dof_len>(dof_beg));
        }
      });
    return ret;
  }
};


template<typename Scalar, typename Storage, template<typename> typename ... Gs>
struct map_trait<BundleBase<Scalar, Storage, Gs...>>
{
  using G = BundleBase<Scalar, Storage, Gs...>;
  using type = BundleBase<Scalar, MappedStorage<typename G::Scalar, G::lie_size>, Gs ...>;
  using const_type = BundleBase<Scalar, const MappedStorage<typename G::Scalar, G::lie_size>, Gs ...>;
};

template<typename _Scalar, template<typename> typename ... _Gs>
using Bundle = BundleBase<
  _Scalar,
  DefaultStorage<_Scalar, iseq_sum<std::index_sequence<lie_info<_Gs<_Scalar>>::lie_size ...>>::value>,
  _Gs...
>;

}  // namespace smooth

#endif  // SMOOTH__BUNDLE_HPP_
