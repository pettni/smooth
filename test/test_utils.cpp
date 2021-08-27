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

#include "smooth/internal/utils.hpp"

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
