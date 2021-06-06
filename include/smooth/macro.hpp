#ifndef SMOOTH__MACRO_HPP_
#define SMOOTH__MACRO_HPP_

namespace smooth {

// Boilerplate that is common for all groups and the bundle
#define SMOOTH_BOILERPLATE(X)                                                              \
  using Storage = _Storage;                                                                \
  using Scalar  = _Scalar;                                                                 \
                                                                                           \
  using Tangent     = Eigen::Matrix<Scalar, Dof, 1>;                                       \
  using TangentMap  = Eigen::Matrix<Scalar, Dof, Dof>;                                     \
  using Vector      = Eigen::Matrix<Scalar, ActDim, 1>;                                    \
  using MatrixGroup = Eigen::Matrix<Scalar, Dim, Dim>;                                     \
                                                                                           \
  /* required to satisfy manifold concept */                                               \
  static constexpr Eigen::Index SizeAtCompileTime = Dof;                                   \
                                                                                           \
  X()            = default;                                                                \
  X(const X & o) = default;                                                                \
  X(X && o)      = default;                                                                \
  X & operator=(const X & o) = default;                                                    \
  X & operator=(X && o) = default;                                                         \
  ~X()                  = default;                                                         \
  /* degrees of freedom at runtime */                                                      \
  Eigen::Index size() const { return Dof; };                                               \
  /* constructor for storage that takes an argument */                                     \
  template<typename S>                                                                     \
  explicit X(S && s) requires std::is_constructible_v<Storage, S> : s_(std::forward<S>(s)) \
  {                                                                                        \
  }

// Boilerplate to be used for regular groups
#define SMOOTH_GROUP_BOILERPLATE(X)                                              \
  SMOOTH_BOILERPLATE(X)                                                          \
  using PlainObject = X<Scalar, DefaultStorage<Scalar, RepSize>>;                \
  /* copy constructor from other storage */                                      \
  template<StorageLike OS>                                                       \
  X(const X<Scalar, OS> & o)                                                     \
  requires ModifiableStorageLike<Storage>                                        \
  {                                                                              \
    meta::static_for<RepSize>([&](auto i) { s_[i] = o.coeffs()[i]; });           \
  }                                                                              \
  /* copy assignment from other storage */                                       \
  template<StorageLike OS>                                                       \
  X & operator=(const X<Scalar, OS> & o) requires ModifiableStorageLike<Storage> \
  {                                                                              \
    meta::static_for<RepSize>([&](auto i) { s_[i] = o.s_[i]; });                 \
    return *this;                                                                \
  }                                                                              \
  /* friend with base */                                                         \
  friend class LieGroupBase<X<Scalar, Storage>, RepSize>;                        \
  /* friend with other storage types */                                          \
  template<typename OtherScalar, StorageLike OS>                                 \
  requires std::is_same_v<Scalar, OtherScalar> friend class X;

// Bundle requires slightly different macro due to more template args
#define SMOOTH_BUNDLE_BOILERPLATE(X)                                                        \
  SMOOTH_BOILERPLATE(X)                                                                     \
  using PlainObject = X<Scalar, DefaultStorage<Scalar, RepSize>, _Gs...>;                   \
  /* copy constructor from other storage */                                                 \
  template<StorageLike OS>                                                                  \
  X(const X<Scalar, OS, _Gs...> & o)                                                        \
  requires ModifiableStorageLike<Storage>                                                   \
  {                                                                                         \
    meta::static_for<RepSize>([&](auto i) { s_[i] = o.coeffs()[i]; });                      \
  }                                                                                         \
  /* copy assignment from other storage */                                                  \
  template<StorageLike OS>                                                                  \
  X & operator=(const X<Scalar, OS, _Gs...> & o) requires ModifiableStorageLike<Storage>    \
  {                                                                                         \
    meta::static_for<RepSize>([&](auto i) { s_[i] = o.s_[i]; });                            \
    return *this;                                                                           \
  }                                                                                         \
  /* friend with base */                                                                    \
  friend class LieGroupBase<X<Scalar, Storage, _Gs...>, RepSize>;                           \
  /* friend with other storage types */                                                     \
  template<typename OtherScalar, MappableStorageLike OS, template<typename> typename... Gs> \
  requires std::is_same_v<Scalar, OtherScalar> friend class X;

}  // namespace smooth

#endif  // SMOOTH__MACRO_HPP_
