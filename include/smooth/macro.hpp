#ifndef SMOOTH__MACRO_HPP_
#define SMOOTH__MACRO_HPP_

namespace smooth{

// Boilerplate that is common for all groups and the bundle
#define SMOOTH_BOILERPLATE(X) \
  using Storage = _Storage;                                            \
  using Scalar = _Scalar;                                              \
                                                                       \
  using Tangent = Eigen::Matrix<Scalar, lie_dof, 1>;                   \
  using TangentMap = Eigen::Matrix<Scalar, lie_dof, lie_dof>;          \
  using Vector = Eigen::Matrix<Scalar, lie_actdim, 1>;                 \
  using MatrixGroup = Eigen::Matrix<Scalar, lie_dim, lie_dim>;         \
                                                                       \
  X() = default;                                                       \
  X(const X & o) = default;                                            \
  X(X && o) = default;                                                 \
  X & operator=(const X & o) = default;                                \
  X & operator=(X && o) = default;                                     \
  ~X() = default;                                                      \
  /* constructor for storage that takes an argument */                 \
  template<typename S>                                                 \
  explicit X(S && s) requires std::is_constructible_v<Storage, S>      \
  : s_(std::forward<S>(s)) {}

// Boilerplate to be used for regular groups
#define SMOOTH_GROUP_BOILERPLATE(X)                                    \
  SMOOTH_BOILERPLATE(X)                                                \
  using Group = X<Scalar, DefaultStorage<Scalar, lie_size>>;           \
  /* copy constructor from other storage */                            \
  template<StorageLike OS>                                             \
  X(const X<Scalar, OS> & o) requires ModifiableStorageLike<Storage>   \
  { meta::static_for<lie_size>([&](auto i) {s_[i] = o.coeffs()[i];});} \
  /* copy assignment from other storage */                             \
  template<StorageLike OS>                                             \
  X & operator=(const X<Scalar, OS> & o)                               \
  requires ModifiableStorageLike<Storage>                              \
  {                                                                    \
    meta::static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});        \
    return *this;                                                      \
  }                                                                    \
  /* friend with base */                                               \
  friend class LieGroupBase<X<Scalar, Storage>, lie_size>;             \
  /* friend with other storage types */                                \
  template<typename OtherScalar, StorageLike OS>                       \
  requires std::is_same_v<Scalar, OtherScalar>                         \
  friend class X;

// Bundle requires slightly different macro due to more template args
#define SMOOTH_BUNDLE_BOILERPLATE(X)                                   \
  SMOOTH_BOILERPLATE(X)                                                \
  using Group = X<Scalar, DefaultStorage<Scalar, lie_size>, _Gs...>;   \
  /* copy constructor from other storage */                            \
  template<StorageLike OS>                                             \
  X(const X<Scalar, OS, _Gs...> & o)                                   \
  requires ModifiableStorageLike<Storage>                              \
  { meta::static_for<lie_size>([&](auto i) {s_[i] = o.coeffs()[i];});} \
  /* copy assignment from other storage */                             \
  template<StorageLike OS>                                             \
  X & operator=(const X<Scalar, OS, _Gs...> & o)                       \
  requires ModifiableStorageLike<Storage>                              \
  {                                                                    \
    meta::static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});        \
    return *this;                                                      \
  }                                                                    \
  /* friend with base */                                               \
  friend class LieGroupBase<X<Scalar, Storage, _Gs...>, lie_size>;     \
  /* friend with other storage types */                                \
  template<                                                            \
    typename OtherScalar,                                              \
    MappableStorageLike OS,                                            \
    template<typename> typename ... Gs                                 \
  >                                                                    \
  requires std::is_same_v<Scalar, OtherScalar>                         \
  friend class X;
}  // namespace smooth

#endif  // SMOOTH__MACRO_HPP_
