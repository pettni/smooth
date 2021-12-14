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

#include <gtest/gtest.h>

#include <sstream>

#include "smooth/manifold_vector.hpp"
#include "smooth/optim.hpp"
#include "smooth/so3.hpp"

template<smooth::Manifold M>
void test(const M &)
{}

TEST(ManifoldVector, Static)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  test(m);
}

TEST(ManifoldVector, Construct)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m1, m2;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());
  m2.push_back(smooth::SO3d::Random());

  auto log = m2 - m1;

  ASSERT_EQ(log.size(), 9);
}

TEST(ManifoldVector, Cast)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m1;

  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());
  m1.push_back(smooth::SO3d::Random());

  auto m1_cast = m1.cast<float>();

  ASSERT_EQ(m1_cast.size(), 9);
}

TEST(ManifoldVector, Optimize)
{
  auto f = []<typename T>(const smooth::ManifoldVector<smooth::SO3<T>> & var) -> Eigen::Vector3<T> {
    Eigen::Vector3<T> ret;
    ret.setZero();
    for (const auto & gi : var) { ret += gi.log().cwiseAbs2(); }
    return ret;
  };

  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  smooth::minimize(f, smooth::wrt(m));

  for (auto x : m) { ASSERT_LE(x.log().norm(), 1e-5); }
}

TEST(ManifoldVector, print)
{
  using M = smooth::ManifoldVector<smooth::SO3d>;
  M m;

  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());
  m.push_back(smooth::SO3d::Random());

  std::stringstream ss;
  ss << m << '\n';
  ASSERT_GE(ss.str().size(), 0);
}
