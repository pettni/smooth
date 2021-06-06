#ifndef SMOOTH__LIE_VECTOR_HPP_
#define SMOOTH__LIE_VECTOR_HPP_

#include <Eigen/Sparse>
#include <numeric>

#include "concepts.hpp"

namespace smooth {

template<typename _Scalar, template<typename> typename _M>
requires Manifold<_M<_Scalar>>
class ManifoldVector
    : public std::vector<_M<_Scalar>, Eigen::aligned_allocator<_M<_Scalar>>> {
private:
  using U    = _M<_Scalar>;
  using Base = std::vector<U, Eigen::aligned_allocator<U>>;

public:
  static constexpr Eigen::Index SizeAtCompileTime = -1;

  using PlainObject = ManifoldVector<_Scalar, _M>;

  using Scalar = _Scalar;

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
  ManifoldVector(Ts &&... ts) : std::vector<_M<_Scalar>>(std::forward<Ts>(ts)...)
  {
  }

  /**
   * @brief Cast to different scalar type
   */
  template<typename NewScalar>
  ManifoldVector<NewScalar, _M> cast() const
  {
    ManifoldVector<NewScalar, _M> ret(vector_size());
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
    if constexpr (U::SizeAtCompileTime > 0) {
      return vector_size() * U::Dof;
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
      this->operator[](i) += a.template segment<U::SizeAtCompileTime>(idx, size_i);
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
  Eigen::Matrix<_Scalar, -1, 1> operator-(const PlainObject & o) const
  {
    std::size_t dof = 0;
    if (U::SizeAtCompileTime > 0) {
      dof = U::SizeAtCompileTime * vector_size();
    } else {
      for (auto i = 0u; i != vector_size(); ++i) { dof += this->operator[](i).size(); }
    }

    Eigen::Matrix<_Scalar, -1, 1> ret(dof);
    Eigen::Index idx = 0;
    for (auto i = 0u; i != vector_size(); ++i) {
      const auto & size_i                                     = this->operator[](i).size();
      ret.template segment<U::SizeAtCompileTime>(idx, size_i) = this->operator[](i) - o[i];
      idx += size_i;
    }

    return ret;
  }
};

}  // namespace smooth

template<typename Stream, typename Scalar, template<typename> typename... _M>
Stream & operator<<(Stream & s, const smooth::ManifoldVector<Scalar, _M...> & g)
{
  s << "ManifoldVector with " << g.vector_size() << " elements:" << std::endl;
  for (auto i = 0u; i != g.vector_size(); ++i) {
    s << i << ": " << g[i];
    if (i != g.vector_size() - 1) { s << std::endl; }
  }
  return s;
}

#endif  // SMOOTH__LIE_VECTOR_HPP_
