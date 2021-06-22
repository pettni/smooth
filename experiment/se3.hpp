#ifndef SE3_HPP_
#define SE3_HPP_

#include <Eigen/Core>

#include <complex>

#include "so3.hpp"
#include "impl/se3.hpp"
#include "lie_group.hpp"

namespace smooth {

// CRTP BASE

template<typename Derived>
class SE3Base : public LieGroup<Derived>
{
protected:
  using Base = LieGroup<Derived>;
  SE3Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  /**
   * Access SO(3) part
   */
  Eigen::Map<SO3<Scalar>> so3()
  {
    return Eigen::Map<SO3<Scalar>>(Base::data() + 3);
  }

  /**
   * Const access SO(3) part
   */
  Eigen::Map<const SO3<Scalar>> so3() const
  {
    return Eigen::Map<const SO3<Scalar>>(Base::data() + 3);
  }

  /**
   * Access T(3) part
   */
  Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> t3()
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 3, 1>>(Base::data());
  }

  /**
   * Const access T(3) part
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t3() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(Base::data());
  }

  /**
   * Tranformation action on 3D vector
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v)
  {
    return so3().matrix() * v + t3();
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SE3;

template<typename _Scalar>
struct lie_traits<SE3<_Scalar>>
{
  using Impl   = SE3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE3<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SE3 : public SE3Base<SE3<_Scalar>>
{
  using Base = typename SE3Base<SE3<_Scalar>>::Base;
  SMOOTH_GROUP_API(SE3)
public:
  /**
   * @brief Construct from SO2 and translation
   */
  template<typename SO3Derived, typename T3Derived>
  SE3(const SO3Base<SO3Derived> & so3, const Eigen::MatrixBase<T3Derived> & t3)
  {
    using std::cos, std::sin;
    so3() = so3;
    t3() = t3;
  }
};

}  // namespace smooth

// MAP TYPE TRAITS

template<typename Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SE3<Scalar>>> : public lie_traits<smooth::SE3<Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SE3<_Scalar>> : public smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>
{
  using Base = typename smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>::Base;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SE3<Scalar>>> : public lie_traits<smooth::SE3<Scalar>>
{};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SE3<_Scalar>> : public smooth::SE3Base<Eigen::Map<const smooth::SE3<_Scalar>>>
{
  using Base = typename smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>::Base;
  SMOOTH_CONST_MAP_API(Map)
};

#endif  // SE3_HPP_
