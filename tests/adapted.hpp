// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include "smooth/lie_group.hpp"

template<typename Scalar>
struct MyGroup
{
  Scalar data;
};

template<typename _Scalar>
struct smooth::traits::lie<MyGroup<_Scalar>>
{
  using Scalar      = _Scalar;
  using PlainObject = MyGroup<Scalar>;
  template<typename NewScalar>
  using CastT = MyGroup<NewScalar>;

  static constexpr int Dof            = 1;
  static constexpr int Dim            = 2;
  static constexpr bool IsCommutative = false;

private:
  using Matrix     = Eigen::Matrix<Scalar, Dim, Dim>;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;
  using Hessian    = Eigen::Matrix<Scalar, Dof, Dof * Dof>;

public:
  // group interface

  static PlainObject Identity([[maybe_unused]] Eigen::Index dof)
  {
    assert(dof == 1);
    return PlainObject{0};
  }
  static PlainObject Random([[maybe_unused]] Eigen::Index dof)
  {
    assert(dof == 1);
    return PlainObject(
      Scalar(-1) + static_cast<Scalar>(rand()) / static_cast<Scalar>(RAND_MAX / 2));
  }
  static TangentMap Ad(PlainObject) { return TangentMap{1}; }
  static PlainObject composition(PlainObject g1, PlainObject g2)
  {
    return PlainObject{.data = g1.data + g2.data};
  }
  static Eigen::Index dof(PlainObject) { return 1; }
  static Eigen::Index dim(PlainObject) { return 2; }
  static PlainObject inverse(PlainObject g) { return PlainObject{.data = -g.data}; }
  static bool isApprox(PlainObject g1, PlainObject g2, Scalar eps)
  {
    using std::abs;
    return abs<Scalar>(g1.data - g2.data) <= eps * abs<Scalar>(g1.data);
  }
  static Tangent log(PlainObject g) { return Tangent{g.data}; }
  static Matrix matrix(PlainObject g) { return Eigen::Matrix2<Scalar>{{{1, g.data}, {0, 1}}}; }
  template<typename NewScalar>
  static MyGroup<NewScalar> cast(PlainObject g)
  {
    return MyGroup<NewScalar>(g.data);
  }

  // tangent interface

  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Zero();
  }
  template<typename Derived>
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
  {
    return PlainObject{a(0)};
  }
  template<typename Derived>
  static Matrix hat(const Eigen::MatrixBase<Derived> & a)
  {
    return Eigen::Matrix2<Scalar>{
      {0, a(0)},
      {0, 0},
    };
  }
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  {
    return Tangent{A(0, 1)};
  }
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return TangentMap::Identity();
  }
  template<typename Derived>
  static TangentMap d2r_exp(const Eigen::MatrixBase<Derived> &)
  {
    return Hessian::Zero();
  }
  template<typename Derived>
  static TangentMap d2r_expinv(const Eigen::MatrixBase<Derived> &)
  {
    return Hessian::Zero();
  }
};
