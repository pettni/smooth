#ifndef SMOOTH__MANIFOLD_VECTOR_HPP_
#define SMOOTH__MANIFOLD_VECTOR_HPP_

#include <Eigen/Sparse>
#include <numeric>

#include "concepts.hpp"

namespace smooth {

/**
 * \p std::vector based container to treat a collection
 * \f[
 *   m = (m_1, m_2, ..., m_k) \in M \times M \times ... \times M
 * \f]
 * of Manifold elements as a single Manifold element \f$m\f$.
 *
 * \warning calling size() on a ManifoldVector returns
 * the degrees of freedom of M x M x ... x M, NOT the
 * number of elements in the vector. For the latter,
 * call vector_size().
 */
template<Manifold M, template<typename> typename Allocator = std::allocator>
class ManifoldVector : public std::vector<M, Allocator<M>>
{
private:
  using Base = std::vector<M, Allocator<M>>;

public:
  //! Degrees of freedom of manifold (equal to tangent space dimentsion)
  static constexpr Eigen::Index SizeAtCompileTime = -1;
  //! Plain return type
  using PlainObject = ManifoldVector<M, Allocator>;
  //! Scalar type
  using Scalar = typename M::Scalar;

  //! Default constructor of empty ManifoldVector
  ManifoldVector()                         = default;
  //! Copy constructor
  ManifoldVector(const ManifoldVector & o) = default;
  //! Move constructor
  ManifoldVector(ManifoldVector && o)      = default;
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
    using CastT = typename decltype(M{}.template cast<NewScalar>())::PlainObject;
    ManifoldVector<CastT, Allocator> ret;
    ret.reserve(vector_size());
    std::transform(this->begin(), this->end(), std::back_insert_iterator(ret), [](const auto & x) {
      return x.template cast<NewScalar>();
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
    if constexpr (M::SizeAtCompileTime > 0) {
      return vector_size() * M::SizeAtCompileTime;
    } else {
      return std::accumulate(this->begin(), this->end(), 0u, [](auto & v1, const auto & item) {
        return v1 + item.size();
      });
    }
  }

  /**
   * @brief In-place addition.
   *
   * \note It must hold that size() == a.size()
   */
  template<typename Derived>
  PlainObject & operator+=(const Eigen::MatrixBase<Derived> & a)
  {
    Eigen::Index idx = 0;
    for (auto i = 0u; i != this->vector_size(); ++i) {
      const auto size_i = this->operator[](i).size();
      this->operator[](i) += a.template segment<M::SizeAtCompileTime>(idx, size_i);
      idx += size_i;
    }
    return *this;
  }

  /**
   * @brief Addition.
   *
   * \note It must hold that size() == a.size()
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
   * \note It must hold that size() == o.size()
   */
  Eigen::Matrix<Scalar, -1, 1> operator-(const PlainObject & o) const
  {
    std::size_t dof = 0;
    if (M::SizeAtCompileTime > 0) {
      dof = M::SizeAtCompileTime * vector_size();
    } else {
      for (auto i = 0u; i != vector_size(); ++i) { dof += this->operator[](i).size(); }
    }

    Eigen::Matrix<Scalar, -1, 1> ret(dof);
    Eigen::Index idx = 0;
    for (auto i = 0u; i != vector_size(); ++i) {
      const auto & size_i                                     = this->operator[](i).size();
      ret.template segment<M::SizeAtCompileTime>(idx, size_i) = this->operator[](i) - o[i];
      idx += size_i;
    }

    return ret;
  }
};

}  // namespace smooth

template<typename Stream, typename M, template<typename> typename Allocator>
Stream & operator<<(Stream & s, const smooth::ManifoldVector<M, Allocator> & g)
{
  s << "ManifoldVector with " << g.vector_size() << " elements:" << std::endl;
  for (auto i = 0u; i != g.vector_size(); ++i) {
    s << i << ": " << g[i];
    if (i != g.vector_size() - 1) { s << std::endl; }
  }
  return s;
}

#endif  // SMOOTH__MANIFOLD_VECTOR_HPP_
