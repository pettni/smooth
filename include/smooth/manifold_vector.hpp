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

#include "adapted_lie_group.hpp"

namespace smooth {

/**
 * @brief \p std::vector based Manifold container.
 *
 * Convenient to treat a collection
 * \f[
 *   m = (m_1, m_2, ..., m_k) \in G \times G \times ... \times G
 * \f]
 * of Manifold elements as a single Manifold element \f$m\f$.
 *
 * @warning calling `size()` on a `ManifoldVector` returns
 * the degrees of freedom of \f$G \times G \times \ldots \times G\f$, NOT the
 * number of elements in the vector. For the latter,
 * use `vector_size()`.
 */
template<AdaptedLieGroup G>
class ManifoldVector : public std::vector<G>
{
private:
  using Base = std::vector<G>;

public:
  //! Degrees of freedom of manifold (equal to tangent space dimentsion)
  static constexpr Eigen::Index SizeAtCompileTime = -1;
  //! Plain return type
  using PlainObject = ManifoldVector<G>;
  //! Scalar type
  using Scalar = ::smooth::Scalar<G>;

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
  auto cast() const
  {
    using CastT = ::smooth::PlainObject<decltype(::smooth::cast<NewScalar>(G{}))>;
    ManifoldVector<CastT> ret;
    ret.reserve(vector_size());
    std::transform(this->begin(), this->end(), std::back_insert_iterator(ret), [](const auto & x) {
      return ::smooth::cast<NewScalar>(x);
    });
    return ret;
  }

  /**
   * @brief Number of elements in ManifoldVector.
   */
  std::size_t vector_size() const { return Base::size(); }

  /**
   * @brief Runtime degrees of freedom.
   *
   * Sum of the degrees of freedom of constituent elements.
   */
  Eigen::Index size() const
  {
    if constexpr (Dof < G >> 0) {
      return vector_size() * Dof<G>;
    } else {
      return std::accumulate(this->begin(), this->end(), 0u, [](auto & v1, const auto & item) {
        return v1 + item.size();
      });
    }
  }

  /**
   * @brief In-place addition.
   *
   * @note It must hold that size() == a.size()
   */
  template<typename Derived>
  PlainObject & operator+=(const Eigen::MatrixBase<Derived> & a)
  {
    Eigen::Index idx = 0;
    for (auto i = 0u; i != this->vector_size(); ++i) {
      const auto size_i   = ::smooth::dof(this->operator[](i));
      this->operator[](i) = rplus(this->operator[](i), a.template segment<Dof<G>>(idx, size_i));
      idx += size_i;
    }
    return *this;
  }

  /**
   * @brief Addition.
   *
   * @note It must hold that `size() == a.size()`
   */
  template<typename Derived>
  PlainObject operator+(const Eigen::MatrixBase<Derived> & a) const
  {
    PlainObject ret = *this;
    ret += a;
    return ret;
  }

  /**
   * @brief Subtraction.
   *
   * @note It must hold that `size() == o.size()`
   */
  Eigen::Matrix<Scalar, -1, 1> operator-(const PlainObject & o) const
  {
    std::size_t dof = 0;
    if (Dof < G >> 0) {
      dof = Dof<G> * vector_size();
    } else {
      for (auto i = 0u; i != vector_size(); ++i) { dof += ::smooth::dof(this->operator[](i)); }
    }

    Eigen::Matrix<Scalar, -1, 1> ret(dof);
    Eigen::Index idx = 0;
    for (auto i = 0u; i != vector_size(); ++i) {
      const auto & size_i                       = ::smooth::dof(this->operator[](i));
      ret.template segment<Dof<G>>(idx, size_i) = rsub(this->operator[](i), o[i]);
      idx += size_i;
    }

    return ret;
  }
};

}  // namespace smooth

template<typename Stream, typename G>
Stream & operator<<(Stream & s, const smooth::ManifoldVector<G> & g)
{
  s << "ManifoldVector with " << g.vector_size() << " elements:" << std::endl;
  for (auto i = 0u; i != g.vector_size(); ++i) {
    s << i << ": " << g[i];
    if (i != g.vector_size() - 1) { s << std::endl; }
  }
  return s;
}

#endif  // SMOOTH__MANIFOLD_VECTOR_HPP_
