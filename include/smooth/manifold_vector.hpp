#ifndef SMOOTH__LIE_VECTOR_HPP_
#define SMOOTH__LIE_VECTOR_HPP_

#include <Eigen/Sparse>
#include <numeric>

#include "concepts.hpp"

namespace smooth {

// TODO ugly with different size()
template<Manifold M, template<typename> typename Allocator = std::allocator>
class ManifoldVector : public std::vector<M, Allocator<M>> {
private:
  using Base = std::vector<M, Allocator<M>>;

public:
  static constexpr Eigen::Index SizeAtCompileTime = -1;

  using PlainObject = ManifoldVector<M, Allocator>;

  using Scalar = typename M::Scalar;

  ManifoldVector()                         = default;
  ManifoldVector(const ManifoldVector & o) = default;
  ManifoldVector(ManifoldVector && o)      = default;
  ManifoldVector & operator=(const ManifoldVector & o) = default;
  ManifoldVector & operator=(ManifoldVector && o) = default;
  ~ManifoldVector()                               = default;

  /**
   * Forwarding constructor to std::vector
   */
  template<typename... Ts>
  ManifoldVector(Ts &&... ts) : Base(std::forward<Ts>(ts)...)
  {
  }

  /**
   * @brief Cast to different scalar type
   */
  template<typename NewScalar>
  auto cast() const
  {
    using CastT = typename decltype(M{}.template cast<NewScalar>())::PlainObject;
    ManifoldVector<CastT, Allocator> ret(vector_size());
    std::transform(this->begin(), this->end(), std::back_insert_iterator(ret), [](const auto & x) {
      return x.template cast<NewScalar>();
    });
    return ret;
  }

  /**
   * @brief Size of vector
   */
  std::size_t vector_size() const { return Base::size(); }

  /**
   * @brief Runtime degrees of freedom
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
   * @brief In-place addition
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
   * @brief Addition
   */
  template<typename Derived>
  PlainObject operator+(const Eigen::MatrixBase<Derived> & a) const
  {
    PlainObject ret = *this;
    ret += a;
    return ret;
  }

  /**
   * @brief Subtraction
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

#endif  // SMOOTH__LIE_VECTOR_HPP_
