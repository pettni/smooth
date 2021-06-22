#ifndef SMOOTH__MACRO_HPP_
#define SMOOTH__MACRO_HPP_

namespace smooth {

#define SMOOTH_INHERIT_TYPEDEFS                 \
  using Base::is_mutable;                       \
  using Base::Dof;                              \
  using Base::RepSize;                          \
  using Scalar        = typename Base::Scalar;  \
  using Tangent       = typename Base::Tangent; \
  using Base::operator=;                        \
  using Base::operator*;

#define SMOOTH_GROUP_API(X)                              \
public:                                                  \
  SMOOTH_INHERIT_TYPEDEFS                                \
  X()          = default;                                \
  X(const X &) = default;                                \
  X(X &&)      = default;                                \
  X & operator=(const X &) = default;                    \
  X & operator=(X &&) = default;                         \
  ~X()                = default;                         \
  template<typename OtherDerived>                        \
                                                         \
  X(const X##Base<OtherDerived> & o)                     \
  {                                                      \
    coeffs() = o.coeffs();                               \
  }                                                      \
  using Storage = Eigen::Matrix<Scalar, RepSize, 1>;     \
                                                         \
  Storage & coeffs() { return coeffs_; }                 \
  const Storage & coeffs() const { return coeffs_; }     \
                                                         \
private:                                                 \
  Storage coeffs_;

#define SMOOTH_MAP_API(X)                                        \
public:                                                          \
  SMOOTH_INHERIT_TYPEDEFS                                        \
  X(Scalar * p) : coeffs_(p) {}                                  \
  X(const X &) = default;                                        \
  X(X &&)      = default;                                        \
  X & operator=(const X &) = default;                            \
  X & operator=(X &&) = default;                                 \
  ~X()                = default;                                 \
                                                                 \
  using Storage = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>>; \
                                                                 \
  Storage & coeffs() { return coeffs_; }                         \
  const Storage & coeffs() const { return coeffs_; }             \
                                                                 \
private:                                                         \
  Storage coeffs_;

#define SMOOTH_CONST_MAP_API(X)                                        \
public:                                                                \
  SMOOTH_INHERIT_TYPEDEFS                                              \
  X(const Scalar * p) : coeffs_(p) {}                                  \
  X(const X &) = default;                                              \
  X(X &&)      = default;                                              \
  X & operator=(const X &) = default;                                  \
  X & operator=(X &&) = default;                                       \
  ~X()                = default;                                       \
                                                                       \
  using Storage = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>>; \
                                                                       \
  const Storage & coeffs() const { return coeffs_; }                   \
  const Scalar * data() const { return coeffs_.data(); }               \
                                                                       \
private:                                                               \
  Storage coeffs_;

}  // namespace smooth

#endif  // SMOOTH__MACRO_HPP_
