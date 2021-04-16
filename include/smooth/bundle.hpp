#ifndef SMOOTH__BUNDLE_HPP_
#define SMOOTH__BUNDLE_HPP_

#include "concepts.hpp"
#include "common.hpp"
#include "lie_group_base.hpp"

namespace smooth
{

template<typename>
struct iseq_sum {};

template<std::size_t ... _Idx>
struct iseq_sum<std::index_sequence<_Idx...>>
{
  static constexpr std::size_t value = (_Idx + ... + 0);
};


template<std::size_t, typename>
struct iseq_el {};

template<std::size_t _Beg, std::size_t ... _Idx>
struct iseq_el<0, std::index_sequence<_Beg, _Idx...>>
{
  static constexpr std::size_t value = _Beg;
};
template<std::size_t _I, std::size_t _Beg, std::size_t ... _Idx>
struct iseq_el<_I, std::index_sequence<_Beg, _Idx...>>
  : public iseq_el<_I - 1, std::index_sequence<_Idx...>>
{};

template<std::size_t _I, typename _Seq>
static constexpr std::size_t iseq_el_v = iseq_el<_I, _Seq>::value;


template<typename>
struct iseq_len {};

template<std::size_t ... _Idx>
struct iseq_len<std::index_sequence<_Idx...>>
{
  static constexpr std::size_t value = sizeof...(_Idx);
};

/**
 * @brief prefix-sum an intseq
 */
template<typename _Collected, typename _Remaining, std::size_t Sum>
struct iseq_psum_impl;

template<std::size_t... _Cur, std::size_t _Sum>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<>, _Sum>
{
  using type = std::index_sequence<_Cur...>;
};

template<std::size_t _First, std::size_t _Sum, std::size_t... _Cur, std::size_t... _Rem>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<_First, _Rem...>, _Sum>
  : public iseq_psum_impl<std::index_sequence<_Cur..., _Sum>, std::index_sequence<_Rem...>,
    _Sum + _First>
{};

template<typename _Seq>
using iseq_psum = iseq_psum_impl<std::index_sequence<>, _Seq, 0>;


template<LieGroupLike ... _Gs>
struct bundle_traits
{
  using lie_sizes = std::index_sequence<_Gs::lie_size ...>;
  using lie_dofs = std::index_sequence<_Gs::lie_dof ...>;
  using lie_dims = std::index_sequence<_Gs::lie_dim ...>;
  using lie_actdims = std::index_sequence<_Gs::lie_actdim ...>;

  static constexpr std::size_t lie_size = iseq_sum<lie_sizes>::value;
  static constexpr std::size_t lie_dof = iseq_sum<lie_dofs>::value;
  static constexpr std::size_t lie_dim = iseq_sum<lie_dims>::value;
  static constexpr std::size_t lie_actdim = iseq_sum<lie_actdims>::value;

  using lie_sizes_psum = typename iseq_psum<lie_sizes>::type;
  using lie_dofs_psum = typename iseq_psum<lie_dofs>::type;
  using lie_dims_psum = typename iseq_psum<lie_dims>::type;
  using lie_actdims_psum = typename iseq_psum<lie_actdims>::type;
};


template<
  typename _Scalar,
  typename _Storage,
  template<typename> typename ... _Gs
>
requires StorageLike<_Storage, _Scalar, bundle_traits<_Gs<_Scalar>...>::lie_size>
struct Bundle
{
private:
  _Storage s_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename OtherScalar, typename OS, template<typename> typename ... Gs>
  requires std::is_same_v<_Scalar, OtherScalar>
  friend class Bundle;

  friend class LieGroupBase<Bundle<_Scalar, _Storage, _Gs...>,
      bundle_traits<_Gs<_Scalar>...>::lie_size>;

  using lie_sizes = bundle_traits<_Gs<_Scalar>...>::lie_sizes;
  using lie_dofs = bundle_traits<_Gs<_Scalar>...>::lie_dofs;
  using lie_dims = bundle_traits<_Gs<_Scalar>...>::lie_dims;
  using lie_actdims = bundle_traits<_Gs<_Scalar>...>::lie_actdims;
  using lie_sizes_psum = bundle_traits<_Gs<_Scalar>...>::lie_sizes_psum;
  using lie_dofs_psum = bundle_traits<_Gs<_Scalar>...>::lie_dofs_psum;
  using lie_dims_psum = bundle_traits<_Gs<_Scalar>...>::lie_dims_psum;
  using lie_actdims_psum = bundle_traits<_Gs<_Scalar>...>::lie_actdims_psum;

public:
  static constexpr uint32_t lie_size = bundle_traits<_Gs<_Scalar>...>::lie_size;
  static constexpr uint32_t lie_dof = bundle_traits<_Gs<_Scalar>...>::lie_dof;
  static constexpr uint32_t lie_dim = bundle_traits<_Gs<_Scalar>...>::lie_dim;
  static constexpr uint32_t lie_actdim = bundle_traits<_Gs<_Scalar>...>::lie_actdim;

  using Scalar = _Scalar;
  using Storage = _Storage;

  using Group = Bundle<Scalar, DefaultStorage<Scalar, lie_size>, _Gs...>;
  using Tangent = Eigen::Matrix<Scalar, lie_dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, lie_dof, lie_dof>;
  using Vector = Eigen::Matrix<Scalar, lie_actdim, 1>;
  using MatrixGroup = Eigen::Matrix<Scalar, lie_dim, lie_dim>;

  // Bundle

  template<std::size_t Idx>
  using PartType = std::tuple_element_t<Idx, std::tuple<_Gs<_Scalar>...>>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  Bundle() = default;
  Bundle(const Bundle & o) = default;
  Bundle(Bundle && o) = default;
  Bundle & operator=(const Bundle & o) = default;
  Bundle & operator=(Bundle && o) = default;
  ~Bundle() = default;

  /**
   * @brief Copy constructor from other storage types
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, lie_size>
  Bundle(const Bundle<Scalar, OS, _Gs...> & o)
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.coeffs()[i];});
  }

  /**
   * @brief Forwarding constructor to storage for map types
   */
  explicit Bundle(Scalar * ptr)
  requires std::is_constructible_v<Storage, Scalar *>
  : s_(ptr) {}

  /**
   * @brief Forwarding constructor to storage for const map types
   */
  explicit Bundle(const Scalar * ptr)
  requires std::is_constructible_v<Storage, const Scalar *>
  : s_(ptr) {}

  /**
   * @brief Copy assignment from other Bundle
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, lie_size>
  Bundle & operator=(const Bundle<Scalar, OS, _Gs...> & o)
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  // BUNDLE-SPECIFIC API

  /**
   * @brief TODO: Construct from components
   */
  // Bundle(const _Gs<_Scalar> ... &)
  // {

  // }

  // Put these here for now since base doesn't support
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

  /**
   * @brief Access parts via map
   */
  template<std::size_t I>
  Map<PartType<I>> part()
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

  void setIdentity() requires ModifiableStorageLike<Storage, Scalar, lie_size>
  {
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t lie_beg = iseq_el_v<i, lie_sizes_psum>;
        Map<PartType<i>>(s_.data() + lie_beg).setIdentity();
      });
  }

  template<typename RNG>
  void setRandom(RNG & rng) requires ModifiableStorageLike<Storage, Scalar, lie_size>
  {
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t lie_beg = iseq_el_v<i, lie_sizes_psum>;
        Map<PartType<i>>(s_.data() + lie_beg).setRandom(rng);
      });
  }

  template<typename NewScalar>
  Bundle<NewScalar, DefaultStorage<NewScalar, lie_size>, _Gs...> cast() const
  {
    Bundle<NewScalar, DefaultStorage<NewScalar, lie_size>, _Gs...> ret;
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
   *
   * Only available for ordered storage
   */
  const Scalar * data() const requires OrderedStorageLike<Storage, Scalar, lie_size>
  {
    return coeffs().data();
  }

  /**
   * @brief Access raw data pointer
   *
   * Only available for ordered modifiable storage
   */
  Scalar * data() requires OrderedModifiableStorageLike<Storage, Scalar, lie_size>
  {
    return coeffs().data();
  }

  /**
   * @brief Overload operator*= for inplace composition
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, lie_size>
  Bundle &operator*=(const Bundle<Scalar, OS, _Gs...> & o)
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
  Bundle & operator+=(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return *this *= exp(t);
  }

  /**
   * @brief Overload operator- for right-minus
   *
   * g1 - g2 := (g2.inverse() * g1).log()
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, lie_size>
  auto operator-(const Bundle<Scalar, OS, _Gs...> & o) const
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

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    MatrixGroup ret;
    ret.setZero();
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        static constexpr std::size_t lie_beg = iseq_el_v<i, lie_sizes_psum>;
        static constexpr std::size_t dim_beg = iseq_el_v<i, lie_dims_psum>;
        static constexpr std::size_t dim_len = iseq_el_v<i, lie_dims>;
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg) =
        ConstMap<PartType<i>>(s_.data() + lie_beg).matrix_group();
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
        ret.template segment<actdim_len>(actdim_beg) =
        part<i>() * x.template segment<actdim_len>(actdim_beg);
      });
    return ret;
  }

  /**
   * @brief Group composition
   */
  template<typename OS, template<typename> typename ... _OGs>
  Group operator*(const Bundle<Scalar, OS, _OGs...> & r) const
  {
    Group ret;
    static_for<sizeof...(_Gs)>(
      [&](auto i) {
        ret.template part<i>() = part<i>() * r.template part<i>();
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
        ret.template part<i>() = part<i>().inverse();
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
        ret.template segment<dof_len>(dof_beg) = part<i>().log();
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
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg) = part<i>().Ad();
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
        ret.template part<i>() = PartType<i>::exp(a.template segment<dof_len>(dof_beg));
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
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg)
          = PartType<i>::ad(a.template segment<dof_len>(dof_beg));
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
        ret.template block<dim_len, dim_len>(dim_beg, dim_beg)
          = PartType<i>::hat(a.template segment<dof_len>(dof_beg));
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
        ret.template segment<dof_len>(dof_beg) = PartType<i>::vee(A.template block<dim_len, dim_len>(dim_beg, dim_beg));
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
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg)
          = PartType<i>::dr_exp(a.template segment<dof_len>(dof_beg));
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
        ret.template block<dof_len, dof_len>(dof_beg, dof_beg)
          = PartType<i>::dr_expinv(a.template segment<dof_len>(dof_beg));
      });
    return ret;
  }
};


template<typename Scalar, typename Storage, template<typename> typename ... Gs>
struct map_trait<Bundle<Scalar, Storage, Gs...>>
{
  using G = Bundle<Scalar, Storage, Gs...>;
  using type = Bundle<Scalar, Eigen::Map<DefaultStorage<typename G::Scalar, G::lie_size>>, Gs ...>;
  using const_type = Bundle<Scalar, Eigen::Map<const DefaultStorage<typename G::Scalar, G::lie_size>>, Gs...>;
};

}  // namespace smooth

#endif  // SMOOTH__BUNDLE_HPP_
