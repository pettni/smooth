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

  static constexpr int Dof = 1;
  static constexpr int Dim = 2;

private:
  using Matrix     = Eigen::Matrix<Scalar, Dim, Dim>;
  using Tangent    = Eigen::Matrix<Scalar, Dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, Dof, Dof>;

public:
  // group interface

  static PlainObject Identity() { return PlainObject{0}; }
  static PlainObject Random()
  {
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
};
