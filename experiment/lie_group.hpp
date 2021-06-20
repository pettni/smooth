#ifndef LIE_GROUP_HPP_
#define LIE_GROUP_HPP_

#include <Eigen/Core>

template<typename Tag>
struct lie_impl;

template<
  typename Scalar,
  typename Tag,
  typename Coeffs = Eigen::Array<Scalar, lie_impl<Tag>::RepSize, 1>
>
class LieGroup {
  using traits = lie_impl<Tag>;

  template<typename __Scalar, typename __Tag, typename __Storage>
  friend class LieGroup;

public:
  static constexpr Eigen::Index RepSize = traits::RepSize;
  static constexpr Eigen::Index Dof     = traits::Dof;

  using Tangent = Eigen::Matrix<Scalar, Dof, 1>;

  // Constructors

  template<typename OtherCoeffs>
  LieGroup(const LieGroup<Scalar, Tag, OtherCoeffs> & o)
  {
    c_ = o.c_;
  }

  template<typename OtherCoeffs>
  LieGroup & operator=(const LieGroup<Scalar, Tag, OtherCoeffs> & o)
  {
    c_ = o.c_;
    return *this;
  }

  // Access coefficients

  Coeffs & coeffs()
  {
    return c_;
  }

  const Coeffs & coeffs() const
  {
    return c_;
  }

  // Group API

  template<typename NewScalar>
  LieGroup<NewScalar, Tag> cast() const
  {
    LieGroup<NewScalar, Tag> ret;
    for (auto i = 0u; i != RepSize; ++i) { ret.c_[i] = static_cast<NewScalar>(c_[i]); }
    return ret;
  }

  Tangent log() const
  {
    Tangent ret;
    traits::log(c_, ret);
    return ret;
  }

  // Tangent API

  template<typename TangentDerived>
  static LieGroup exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    LieGroup<Scalar, Tag> ret;
    traits::exp(a, ret.c_);
    return ret;
  }

protected:
  Coeffs c_;
};

template<typename Stream, typename G>
Stream & operator<<(Stream & s, const G & g)
{
  for (auto i = 0; i != G::RepSize; ++i) {
    s << g.coeffs()[i] << " ";
  }
  return s;
}

#endif // !LIE_GROUP_HPP_
