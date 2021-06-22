#ifndef SO2_HPP_
#define SO2_HPP_

#include <Eigen/Core>

#include <complex>

#include "impl/so2.hpp"
#include "lie_group.hpp"

namespace smooth {

// CRTP BASE

template<typename Derived>
class SO2Base : public LieGroup<Derived>
{
protected:
  using Base = LieGroup<Derived>;
  SO2Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  /**
   * Complex number (U(1)) representation
   */
  std::complex<Scalar> u1() const
  {
    return std::complex<Scalar>(Base::coeffs().y(), Base::coeffs().x());
  }

  /**
   * Angle represetation
   */
  Scalar angle() const
  {
    return Base::log();
  }

  /**
   * Rotation action on 2D vector
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v)
  {
    return Base::matrix() * v;
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SO2;

template<typename _Scalar>
struct lie_traits<SO2<_Scalar>>
{
  using Impl   = SO2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO2<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SO2 : public SO2Base<SO2<_Scalar>>
{
  using Base = typename SO2Base<SO2<_Scalar>>::Base;
  SMOOTH_GROUP_API(SO2)
public:
  /**
   * @brief Construct from complex number
   */
  template<typename Derived>
  SO2(const std::complex<Scalar> & c)
  {
    coeffs_.x() = c.im;
    coeffs_.y() = c.re;
  }

  /**
   * @brief Construct from angle
   */
  template<typename Derived>
  SO2(const Scalar & angle)
  {
    using std::cos, std::sin;
    coeffs_.x() = cos(angle);
    coeffs_.y() = sin(angle);
  }
};

}  // namespace smooth

// MAP TYPE TRAITS

template<typename Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO2<Scalar>>> : public lie_traits<smooth::SO2<Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SO2<_Scalar>> : public smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>
{
  using Base = typename smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>::Base;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SO2<Scalar>>> : public lie_traits<smooth::SO2<Scalar>>
{};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SO2<_Scalar>> : public smooth::SO2Base<Eigen::Map<const smooth::SO2<_Scalar>>>
{
  using Base = typename smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>::Base;
  SMOOTH_CONST_MAP_API(Map)
};

#endif  // SO2_HPP_
