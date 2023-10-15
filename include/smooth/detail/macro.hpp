// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#define SMOOTH_INHERIT_TYPEDEFS           \
  using Base::is_mutable;                 \
  using Base::Dof;                        \
  using Base::RepSize;                    \
  /*! @brief Scalar type. */              \
  using Scalar = typename Base::Scalar;   \
  /*! @brief Tangent type. */             \
  using Tangent = typename Base::Tangent; \
  using Base::operator=;                  \
  using Base::operator*;                  \
                                          \
  static_assert(true, "")

#define SMOOTH_GROUP_API(X)                                               \
                                                                          \
public:                                                                   \
  SMOOTH_INHERIT_TYPEDEFS;                                                \
  /*! @brief Construct uninitialized object */                            \
  X() = default;                                                          \
  /*! @brief Copy constructor */                                          \
  X(const X &) = default;                                                 \
  /*! @brief Move contructor */                                           \
  X(X &&) = default;                                                      \
  /*! @brief Copy assignment */                                           \
  X & operator=(const X &) = default;                                     \
  /*! @brief Move assignment */                                           \
  X & operator=(X &&) = default;                                          \
  /*! @brief Destructor */                                                \
  ~X() = default;                                                         \
  /*! @brief Copy constructor from other storage type */                  \
  template<typename OtherDerived>                                         \
  X(const X##Base<OtherDerived> & o) noexcept                             \
  {                                                                       \
    coeffs() = static_cast<const OtherDerived &>(o).coeffs();             \
  }                                                                       \
  /*! @brief Underlying storage is an Eigen matrix */                     \
  using Storage = Eigen::Matrix<Scalar, RepSize, 1>;                      \
  /*! @brief Access underlying Eigen::Matrix */                           \
  inline Storage & coeffs() noexcept { return m_coeffs; }                 \
  /*! @brief Const access underlying Eigen::Matrix */                     \
  inline const Storage & coeffs() const noexcept { return m_coeffs; }     \
  /*! @brief Access raw pointer */                                        \
  inline Scalar * data() noexcept { return m_coeffs.data(); }             \
  /*! @brief Const access raw pointer */                                  \
  inline const Scalar * data() const noexcept { return m_coeffs.data(); } \
                                                                          \
private:                                                                  \
  Storage m_coeffs;                                                       \
                                                                          \
  static_assert(true, "")

#define SMOOTH_MAP_API()                                                  \
                                                                          \
public:                                                                   \
  SMOOTH_INHERIT_TYPEDEFS;                                                \
  /*! @brief Map memory as Lie type */                                    \
  Map(Scalar * p) : m_coeffs(p) {}                                        \
  /*! @brief Copy constructor */                                          \
  Map(const Map &) = default;                                             \
  /*! @brief Move constructor */                                          \
  Map(Map &&) = default;                                                  \
  /*! @brief Copy assignment */                                           \
  Map & operator=(const Map &) = default;                                 \
  /*! @brief Move assignment */                                           \
  Map & operator=(Map &&) = default;                                      \
  /*! @brief Destructor */                                                \
  ~Map() = default;                                                       \
  /*! @brief Underlying storage is Eigen map */                           \
  using Storage = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>>;          \
  /*! @brief Access underlying Eigen::Map */                              \
  inline Storage & coeffs() noexcept { return m_coeffs; }                 \
  /*! @brief Const access underlying Eigen::Map */                        \
  inline const Storage & coeffs() const noexcept { return m_coeffs; }     \
  /*! @brief Access raw pointer */                                        \
  inline Scalar * data() noexcept { return m_coeffs.data(); }             \
  /*! @brief Const access raw pointer */                                  \
  inline const Scalar * data() const noexcept { return m_coeffs.data(); } \
                                                                          \
private:                                                                  \
  Storage m_coeffs;                                                       \
                                                                          \
  static_assert(true, "")

#define SMOOTH_CONST_MAP_API()                                         \
                                                                       \
public:                                                                \
  SMOOTH_INHERIT_TYPEDEFS;                                             \
  /*! @brief Const map memory as Lie type */                           \
  Map(const Scalar * p) : m_coeffs(p) {}                               \
  /*! @brief Destructor */                                             \
  ~Map() = default;                                                    \
  /*! @brief Underlying storage is Eigen const map */                  \
  using Storage = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>>; \
  /*! @brief Const access underlying Eigen::Map */                     \
  const Storage & coeffs() const { return m_coeffs; }                  \
  /*! @brief Const access raw pointer */                               \
  const Scalar * data() const { return m_coeffs.data(); }              \
                                                                       \
private:                                                               \
  Storage m_coeffs;                                                    \
                                                                       \
  static_assert(true, "")
