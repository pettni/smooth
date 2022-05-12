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

#ifndef SMOOTH__MANIFOLD_VECTOR_HPP_
#define SMOOTH__MANIFOLD_VECTOR_HPP_

#include <Eigen/Sparse>
#include <numeric>

#include "manifold.hpp"

namespace smooth {

/**
 * @brief \p std::vector based Manifold container.
 *
 * Convenient to treat a collection
 * \f[
 *   m = (m_1, m_2, ..., m_k) \in M \times M \times ... \times M
 * \f]
 * of Manifold elements as a single Manifold element \f$m\f$.
 */
template<Manifold M>
class ManifoldVector : public std::vector<M>
{
private:
  using Base = std::vector<M>;

public:
  //! Default constructor of empty ManifoldVector
  ManifoldVector() = default;
  //! Copy constructor
  ManifoldVector(const ManifoldVector & o) = default;
  //! Move constructor
  ManifoldVector(ManifoldVector && o) = default;
  //! Copy assignment operator
  ManifoldVector & operator=(const ManifoldVector & o) = default;
  //! Move assignment operator
  ManifoldVector & operator=(ManifoldVector && o) = default;
  ~ManifoldVector()                               = default;

  /**
   * Forwarding constructor to std::vector.
   */
  template<typename... Ts>
  ManifoldVector(Ts &&... ts) : Base(std::forward<Ts>(ts)...)
  {}

  /**
   * @brief Cast to different scalar type.
   */
  template<typename NewScalar>
  ManifoldVector<CastT<NewScalar, M>> cast() const
  {
    ManifoldVector<CastT<NewScalar, M>> ret;
    ret.reserve(size());
    std::transform(this->begin(), this->end(), std::back_insert_iterator(ret), [](const auto & x) {
      return ::smooth::cast<NewScalar>(x);
    });
    return ret;
  }

  /**
   * @brief Number of elements in ManifoldVector.
   */
  std::size_t size() const { return Base::size(); }

  /**
   * @brief Runtime degrees of freedom.
   *
   * Sum of the degrees of freedom of constituent elements.
   */
  Eigen::Index dof() const
  {
    if constexpr (Dof < M >> 0) {
      return size() * Dof<M>;
    } else {
      return std::accumulate(this->begin(), this->end(), 0u, [](auto s, const auto & item) {
        return s + ::smooth::dof<M>(item);
      });
    }
  }

  /**
   * @brief In-place addition.
   *
   * @note It must hold that dof() == a.dof()
   */
  template<typename Derived>
  ManifoldVector<M> & operator+=(const Eigen::MatrixBase<Derived> & a)
  {
    Eigen::Index dof_cntr = 0;
    for (auto i = 0u; i != this->size(); ++i) {
      const auto dof_i = ::smooth::dof<M>(this->operator[](i));
      this->operator[](i) =
        rplus<M>(this->operator[](i), a.template segment<Dof<M>>(dof_cntr, dof_i));
      dof_cntr += dof_i;
    }
    return *this;
  }

  /**
   * @brief Addition.
   *
   * @note It must hold that `dof() == a.dof()`
   */
  template<typename Derived>
  ManifoldVector<M> operator+(const Eigen::MatrixBase<Derived> & a) const
  {
    ManifoldVector<M> ret = *this;
    ret += a;
    return ret;
  }

  /**
   * @brief Subtraction.
   *
   * @note It must hold that `dof() == o.dof()`
   */
  Eigen::VectorX<Scalar<M>> operator-(const ManifoldVector<M> & o) const
  {
    std::size_t dof_cnts = 0;
    if (Dof < M >> 0) {
      dof_cnts = Dof<M> * size();
    } else {
      for (auto i = 0u; i != size(); ++i) { dof_cnts += ::smooth::dof<M>(this->operator[](i)); }
    }

    Eigen::VectorX<Scalar<M>> ret(dof_cnts);
    Eigen::Index idx = 0;
    for (auto i = 0u; i != size(); ++i) {
      const auto & size_i                       = ::smooth::dof<M>(this->operator[](i));
      ret.template segment<Dof<M>>(idx, size_i) = rminus<M>(this->operator[](i), o[i]);
      idx += size_i;
    }

    return ret;
  }
};

/**
 * @brief Manifold interface for ManifoldVector
 */
template<Manifold M>
struct traits::man<ManifoldVector<M>>
{
  // \cond
  using Scalar      = ::smooth::Scalar<M>;
  using PlainObject = ManifoldVector<M>;
  template<typename NewScalar>
  using CastT = ManifoldVector<typename man<M>::template CastT<NewScalar>>;

  static constexpr Eigen::Index Dof = -1;

  static inline Eigen::Index dof(const ManifoldVector<M> & m) { return m.dof(); }

  static inline PlainObject Default(Eigen::Index dof)
  {
    /// @note If underlying M has dynamic size, mdof not uniquely defined
    const Eigen::Index mdof = ::smooth::Dof<M> != -1 ? ::smooth::Dof<M> : 1;
    const Eigen::Index size = dof / mdof;

    return traits::man<M>::PlainObject(size, traits::man<M>::Default(mdof));
  }

  template<typename NewScalar>
  static inline auto cast(const ManifoldVector<M> & m)
  {
    return m.template cast<NewScalar>();
  }

  template<typename Derived>
  static inline ManifoldVector<M>
  rplus(const ManifoldVector<M> & m, const Eigen::MatrixBase<Derived> & a)
  {
    return m + a;
  }

  static inline Eigen::Matrix<Scalar, Dof, 1>
  rminus(const ManifoldVector<M> & m1, const ManifoldVector<M> & m2)
  {
    return m1 - m2;
  }
  // \endcond
};

}  // namespace smooth

template<typename Stream, smooth::Manifold M>
Stream & operator<<(Stream & s, const smooth::ManifoldVector<M> & g)
{
  s << "ManifoldVector with " << g.size() << " elements:\n";
  for (auto i = 0u; i != g.size(); ++i) {
    s << i << ": " << g[i];
    if (i != g.size() - 1) { s << '\n'; }
  }
  return s;
}

#endif  // SMOOTH__MANIFOLD_VECTOR_HPP_
