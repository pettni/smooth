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

#include <algorithm>

#include "smooth/detail/utils.hpp"

TEST(Utils, BinarySearch)
{
  std::vector<double> v1{0, 1, 2, 3, 4, 5, 6};

  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 0), std::ranges::begin(v1));
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 0.5), std::ranges::begin(v1));
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 1), std::ranges::begin(v1) + 1);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 1.5), std::ranges::begin(v1) + 1);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 2), std::ranges::begin(v1) + 2);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 2.5), std::ranges::begin(v1) + 2);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 5), std::ranges::begin(v1) + 5);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 5.5), std::ranges::begin(v1) + 5);

  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 6), std::ranges::begin(v1) + 6);
  ASSERT_EQ(smooth::utils::binary_interval_search(v1, 6.5), std::ranges::begin(v1) + 6);

  ASSERT_EQ(smooth::utils::binary_interval_search(v1, -0.5), std::ranges::end(v1));

  std::vector<double> v2, v3{0};
  ASSERT_EQ(smooth::utils::binary_interval_search(v2, -1), std::ranges::end(v2));
  ASSERT_EQ(smooth::utils::binary_interval_search(v2, 0), std::ranges::end(v2));

  ASSERT_EQ(smooth::utils::binary_interval_search(v3, -1), std::ranges::end(v3));
  ASSERT_EQ(smooth::utils::binary_interval_search(v3, 0), std::ranges::begin(v3));
}

TEST(Utils, BinarySearchUnequal)
{
  std::vector<double> v{0, 1, 2, 3, 4, 500, 501, 502};

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 0), std::ranges::begin(v));
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 0.5), std::ranges::begin(v));

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 1), std::ranges::begin(v) + 1);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 1.5), std::ranges::begin(v) + 1);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 2), std::ranges::begin(v) + 2);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 2.5), std::ranges::begin(v) + 2);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 3), std::ranges::begin(v) + 3);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 3.5), std::ranges::begin(v) + 3);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 4), std::ranges::begin(v) + 4);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 4.5), std::ranges::begin(v) + 4);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 400), std::ranges::begin(v) + 4);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 500), std::ranges::begin(v) + 5);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 500.5), std::ranges::begin(v) + 5);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 501), std::ranges::begin(v) + 6);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 501.5), std::ranges::begin(v) + 6);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, 502), std::ranges::begin(v) + 7);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, 502.5), std::ranges::begin(v) + 7);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, -0.5), std::ranges::end(v));
}

TEST(Utils, BinarySearchString)
{
  std::vector<std::string> v{"b", "c", "e", "f", "j", "x"};

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "b"), std::ranges::begin(v));
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "c"), std::ranges::begin(v) + 1);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "d"), std::ranges::begin(v) + 1);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "e"), std::ranges::begin(v) + 2);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "f"), std::ranges::begin(v) + 3);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "g"), std::ranges::begin(v) + 3);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "h"), std::ranges::begin(v) + 3);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "j"), std::ranges::begin(v) + 4);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "n"), std::ranges::begin(v) + 4);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "p"), std::ranges::begin(v) + 4);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "x"), std::ranges::begin(v) + 5);
  ASSERT_EQ(smooth::utils::binary_interval_search(v, "y"), std::ranges::begin(v) + 5);

  ASSERT_EQ(smooth::utils::binary_interval_search(v, "a"), std::ranges::end(v));
}

TEST(Utils, RangePairwiseTransformConstexpr)
{
  static constexpr std::array<double, 5> vals{1.1, 2.2, 6.7, 4.6, 5.0};
  constexpr auto fun1 = [](auto x) { return x * x; };
  constexpr auto m1   = std::ranges::max(vals | std::views::transform(fun1));

  static_assert(m1 == 6.7 * 6.7);

  constexpr auto fun2 = [](auto x, auto y) { return y - x; };
  constexpr auto m2   = std::ranges::max(vals | smooth::utils::views::pairwise_transform(fun2));

  static_assert(m2 == 4.5);
}

TEST(Utils, RangePairwiseTransform)
{
  const std::vector<double> values{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};
  const auto fun = [](double d1, double d2) -> double { return d2 - d1; };

  const auto diff_values = values | smooth::utils::views::pairwise_transform(fun);

  using R  = decltype(values);
  using Rt = std::decay_t<decltype(diff_values)>;

  static_assert(std::is_same_v<std::ranges::sentinel_t<R>, std::ranges::sentinel_t<Rt>>);
  static_assert(std::ranges::forward_range<Rt>);
  static_assert(std::forward_iterator<std::ranges::iterator_t<Rt>>);

  static_assert(std::ranges::forward_range<const Rt>);
  static_assert(std::forward_iterator<std::ranges::iterator_t<const Rt>>);

  static_assert(std::ranges::bidirectional_range<Rt>);
  static_assert(std::bidirectional_iterator<std::ranges::iterator_t<Rt>>);

  ASSERT_EQ(std::ranges::size(diff_values), std::ranges::size(values) - 1);

  for (auto i = 0u; auto d : diff_values) {
    ASSERT_EQ(d, fun(values[i], values[i + 1]));
    ++i;
  }

  for (auto i = 0u; auto d : diff_values | std::views::drop(1)) {
    ASSERT_EQ(d, fun(values[i + 1], values[i + 2]));
    ++i;
  }

  for (auto i = 0u; auto d : diff_values | std::views::reverse) {
    const auto N = values.size();
    ASSERT_EQ(d, fun(values[N - 2 - i], values[N - 1 - i]));
    ++i;
  }

  {
    auto it = std::ranges::begin(diff_values);
    std::advance(it, 2);

    auto it2 = it;

    ASSERT_DOUBLE_EQ(*it, -3.4);
    ASSERT_EQ(*it, *it2);
    ASSERT_EQ(it, it2);
  }

  {
    auto it = std::ranges::begin(diff_values);
    std::advance(it, 2);
    auto it2 = std::ranges::begin(diff_values);
    std::advance(it2, 2);

    auto it3 = std::move(it2);

    ASSERT_DOUBLE_EQ(*it, -3.4);
    ASSERT_EQ(*it, *it3);
    ASSERT_EQ(it, it3);
  }

  // sized
  for (auto i = 0; i < 10; ++i) {
    auto irange = std::views::iota(2, 2 + i) | smooth::utils::views::pairwise_transform(fun);

    ASSERT_EQ(std::ranges::size(irange), std::max(i - 1, 0));

    for (auto d : irange) { ASSERT_EQ(d, 1); }
  }

  // unsized
  for (auto i = 0; i < 10; ++i) {
    auto irange = std::views::iota(2) | std::views::take_while([&i](auto x) { return x < i; })
                | smooth::utils::views::pairwise_transform(fun);

    for (auto d : irange) { ASSERT_EQ(d, 1); }
  }
}

TEST(Utils, Zip2)
{
  const std::vector<double> vals1{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};
  const std::vector<int> vals2{1, 5, 6, 7, 8, 10};

  ASSERT_EQ(
    std::ranges::size(smooth::utils::zip(vals1, vals2)), std::min(vals1.size(), vals2.size()));

  for (auto i = 0; const auto & [v1, v2] : smooth::utils::zip(vals1, vals2)) {
    ASSERT_EQ(v1, vals1[i]);
    ASSERT_EQ(v2, vals2[i]);
    ++i;
  }

  for (auto i = 0; const auto & [v1, v2] : smooth::utils::zip(vals1, vals1 | std::views::drop(2))) {
    ASSERT_EQ(v1, vals1[i]);
    ASSERT_EQ(v2, vals1[i + 2]);
    ++i;
  }

  const auto minus = [](auto x) { return x - 1; };

  for (auto i = 0;
       const auto & [v1, v2] : smooth::utils::zip(vals1, vals1 | std::views::transform(minus))) {
    ASSERT_EQ(v1, vals1[i]);
    ASSERT_EQ(v2, vals1[i] - 1);
    ++i;
  }
}

TEST(Utils, Zip2Modify)
{
  std::vector<double> vals1{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};
  std::vector<int> vals2{1, 5, 6, 7, 8, 10};

  const auto S = std::ranges::size(smooth::utils::zip(vals1, vals2));
  ASSERT_EQ(S, std::min(vals1.size(), vals2.size()));

  for (auto && [v1, v2] : smooth::utils::zip(vals1, vals2)) {
    v1 = -1;
    v2 = -2;
  }

  for (auto i = 0u; i < S; ++i) {
    ASSERT_EQ(vals1[i], -1);
    ASSERT_EQ(vals2[i], -2);
  }
}

TEST(Utils, Zip3)
{
  const std::vector<double> vals1{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};
  const std::vector<int> vals2{1, 5, 6, 7, 8, 10};
  const std::vector<std::string> vals3{"Hej", "det", "har", "ar", "en", "stringvector"};

  ASSERT_EQ(
    std::ranges::size(smooth::utils::zip(vals1, vals2, vals3)),
    std::min<std::size_t>({vals1.size(), vals2.size(), vals3.size()}));

  for (auto i = 0; const auto & [v1, v2, v3] : smooth::utils::zip(vals1, vals2, vals3)) {
    ASSERT_EQ(v1, vals1[i]);
    ASSERT_EQ(v2, vals2[i]);
    ASSERT_EQ(v3, vals3[i]);
    ++i;
  }
}

TEST(Utils, Zip4Iota)
{
  const std::vector<double> vals1{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};
  const std::vector<int> vals2{1, 5, 6, 7, 8, 10};
  const std::vector<std::string> vals3{"Hej", "det", "har", "ar", "en", "stringvector"};

  ASSERT_EQ(
    std::ranges::size(smooth::utils::zip(vals1, vals2, vals3)),
    std::min<std::size_t>({vals1.size(), vals2.size(), vals3.size()}));

  for (const auto & [i, v1, v2, v3] :
       smooth::utils::zip(std::views::iota(0), vals1, vals2, vals3)) {
    ASSERT_EQ(v1, vals1[i]);
    ASSERT_EQ(v2, vals2[i]);
    ASSERT_EQ(v3, vals3[i]);
  }
}

TEST(Utils, AdjacentDrop)
{
  const std::vector<double> vals1{1, 2.5, 5.5, 2.1, 5, 6, 7, 8, 10};

  const auto fun1 = [](auto x) { return 2 * x; };
  const auto fun2 = [](auto x, auto y) { return 2 * x + y; };

  auto trans_view         = vals1 | std::views::transform(fun1) | std::views::drop(1);
  const auto c_trans_view = vals1 | std::views::transform(fun1) | std::views::drop(1);

  static_assert(std::ranges::range<decltype(trans_view)>);
  static_assert(std::ranges::range<decltype(c_trans_view)>);

  auto pw_view = vals1 | smooth::utils::views::pairwise_transform(fun2);
  static_assert(std::ranges::range<decltype(pw_view)>);

  const auto c_pw_view = vals1 | smooth::utils::views::pairwise_transform(fun2);
  static_assert(std::ranges::range<decltype(c_pw_view)>);

  auto pw_view_drop = pw_view | std::views::drop(1);
  static_assert(std::ranges::range<decltype(pw_view_drop)>);

  auto c_pw_view_drop = c_pw_view | std::views::drop(1);
  static_assert(std::ranges::range<decltype(c_pw_view_drop)>);
}
