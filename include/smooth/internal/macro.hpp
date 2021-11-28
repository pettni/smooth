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

#ifndef SMOOTH__INTERNAL__MACRO_HPP_
#define SMOOTH__INTERNAL__MACRO_HPP_

namespace smooth {

#define SMOOTH_INHERIT_TYPEDEFS                 \
  using Base::is_mutable;                       \
  using Base::Dof;                              \
  using Base::RepSize;                          \
  /*! @brief Scalar type. */                    \
  using Scalar = typename Base::Scalar;         \
  /*! @brief Tangent type. */                   \
  using Tangent       = typename Base::Tangent; \
  using Base::operator=;                        \
  using Base::operator*;                        \
                                                \
  static_assert(true, "")

#define SMOOTH_GROUP_API(X)                                   \
                                                              \
public:                                                       \
  SMOOTH_INHERIT_TYPEDEFS;                                    \
  /*! @brief Construct uninitialized object */                \
  X() = default;                                              \
  /*! @brief Copy constructor */                              \
  X(const X &) = default;                                     \
  /*! @brief Move contructor */                               \
  X(X &&) = default;                                          \
  /*! @brief Copy assignment */                               \
  X & operator=(const X &) = default;                         \
  /*! @brief Move assignment */                               \
  X & operator=(X &&) = default;                              \
  /*! @brief Destructor */                                    \
  ~X() = default;                                             \
  /*! @brief Copy constructor from other storage type */      \
  template<typename OtherDerived>                             \
  X(const X##Base<OtherDerived> & o)                          \
  {                                                           \
    coeffs() = static_cast<const OtherDerived &>(o).coeffs(); \
  }                                                           \
  /*! @brief Underlying storage is an Eigen matrix */         \
  using Storage = Eigen::Matrix<Scalar, RepSize, 1>;          \
  /*! @brief Access underlying Eigen::Matrix */               \
  Storage & coeffs() { return coeffs_; }                      \
  /*! @brief Const access underlying Eigen::Matrix */         \
  const Storage & coeffs() const { return coeffs_; }          \
  /*! @brief Access raw pointer */                            \
  Scalar * data() { return coeffs_.data(); }                  \
  /*! @brief Const access raw pointer */                      \
  const Scalar * data() const { return coeffs_.data(); }      \
                                                              \
private:                                                      \
  Storage coeffs_;                                            \
                                                              \
  static_assert(true, "")

#define SMOOTH_MAP_API()                                         \
                                                                 \
public:                                                          \
  SMOOTH_INHERIT_TYPEDEFS;                                       \
  /*! @brief Map memory as Lie type */                           \
  Map(Scalar * p) : coeffs_(p) {}                                \
  /*! @brief Copy constructor */                                 \
  Map(const Map &) = default;                                    \
  /*! @brief Move constructor */                                 \
  Map(Map &&) = default;                                         \
  /*! @brief Copy assignment */                                  \
  Map & operator=(const Map &) = default;                        \
  /*! @brief Move assignment */                                  \
  Map & operator=(Map &&) = default;                             \
  /*! @brief Destructor */                                       \
  ~Map() = default;                                              \
  /*! @brief Underlying storage is Eigen map */                  \
  using Storage = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>>; \
  /*! @brief Access underlying Eigen::Map */                     \
  Storage & coeffs() { return coeffs_; }                         \
  /*! @brief Const access underlying Eigen::Map */               \
  const Storage & coeffs() const { return coeffs_; }             \
  /*! @brief Access raw pointer */                               \
  Scalar * data() { return coeffs_.data(); }                     \
  /*! @brief Const access raw pointer */                         \
  const Scalar * data() const { return coeffs_.data(); }         \
                                                                 \
private:                                                         \
  Storage coeffs_;                                               \
                                                                 \
  static_assert(true, "")

#define SMOOTH_CONST_MAP_API()                                         \
                                                                       \
public:                                                                \
  SMOOTH_INHERIT_TYPEDEFS;                                             \
  /*! @brief Const map memory as Lie type */                           \
  Map(const Scalar * p) : coeffs_(p) {}                                \
  /*! @brief Destructor */                                             \
  ~Map() = default;                                                    \
  /*! @brief Underlying storage is Eigen const map */                  \
  using Storage = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>>; \
  /*! @brief Const access underlying Eigen::Map */                     \
  const Storage & coeffs() const { return coeffs_; }                   \
  /*! @brief Const access raw pointer */                               \
  const Scalar * data() const { return coeffs_.data(); }               \
                                                                       \
private:                                                               \
  Storage coeffs_;                                                     \
                                                                       \
  static_assert(true, "")

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__MACRO_HPP_
