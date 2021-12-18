// lp2d: Two-Dimensional Linear Programming
// https://github.com/pettni/lp2d
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

#ifndef LP2D__LP2D_HPP_
#define LP2D__LP2D_HPP_

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <vector>

namespace lp2d {

using Scalar = double;

enum class Status { Optimal, PrimaryInfeasible, DualInfeasible };

////////////////////////////////
///// FORWARD DECLARATIONS /////
////////////////////////////////

namespace detail {

inline constexpr auto eps = 100 * std::numeric_limits<Scalar>::epsilon();
inline constexpr auto inf = std::numeric_limits<Scalar>::infinity();

/// @brief Halfplane represented as inequality ax + by <= c
struct HalfPlane
{
  Scalar a, b, c;
  bool active{true};
};

inline std::tuple<Scalar, Scalar, Status> solve_impl(std::vector<HalfPlane> &);

}  // namespace detail

////////////////////////////////
//////// USER INTERFACE ////////
////////////////////////////////

/**
 * @brief Solve 2D linear program
 *
 *  min  cx * x + cy * y
 *  s.t. ax * x + ay * y <= b   for (ax, ay, b) in hps
 *
 * @param cx, cy objective function
 * @param rows triplets (ax, ay, b) defining rows of the LP
 * @return {xopt, yopt} optimal solution
 *
 * If problem is infeasible yopt = inf
 */
template<std::ranges::range R>
std::tuple<Scalar, Scalar, Status> solve(Scalar cx, Scalar cy, const R & rows) requires(
  std::tuple_size_v<std::ranges::range_value_t<R>> == 3)
{
  const Scalar sqnorm = cx * cx + cy * cy;

  if (sqnorm < detail::eps) { return {0, 0, Status::Optimal}; }

  if (std::ranges::empty(rows)) { return {0, 0, Status::DualInfeasible}; }

  // transformation matrix:
  //  [x; y] = [cP -sP; sP cP] [xt; yt]
  const Scalar cP = cy / sqnorm;
  const Scalar sP = -cx / sqnorm;

  // insert rotated halfplanes with unit vector norm 1
  std::vector<detail::HalfPlane> input;
  input.reserve(std::ranges::size(rows));
  for (const auto [a, b, c] : rows) {
    const double ra = cP * a + sP * b;
    const double rb = -sP * a + cP * b;

    const double norm = ra * ra + rb * rb;

    if (norm > detail::eps && c < detail::inf) {
      input.push_back(detail::HalfPlane{
        .a      = ra / norm,
        .b      = rb / norm,
        .c      = c / norm,
        .active = true,
      });
    }
  }

  // scale factor
  Scalar lambda{1};
  for (const auto & hp : input) { lambda = std::max(lambda, std::abs(hp.c)); }

  for (auto & hp : input) { hp.c /= lambda; }

  const auto [xt_opt, yt_opt, status] = solve_impl(input);

  // multiplication that returns 0 for 0 * inf (regular multiplication returns nan)
  const auto mul = [](Scalar a, Scalar b) { return std::abs(a) > detail::eps ? a * b : 0; };

  // return solution in original coordinates
  return {
    lambda * (mul(cP, xt_opt) - mul(sP, yt_opt)),
    lambda * (mul(sP, xt_opt) + mul(cP, yt_opt)),
    status,
  };
}

////////////////////////////////
//////// IMPLEMENTATION ////////
////////////////////////////////

namespace detail
{

  // value and derivative
  using ValDer = std::tuple<Scalar, Scalar>;

  // value and subderivative (upper,lower)
  using ValSubDer = std::tuple<Scalar, Scalar, Scalar>;

  // halfplanes that define upper and lower bounds on y (as a function of x)
  constexpr auto active_y_upper = [](const auto & hp) { return hp.active && hp.b > 0; };
  constexpr auto active_y_lower = [](const auto & hp) { return hp.active && hp.b < 0; };

  // halfplane as bounds on y
  constexpr auto hp_to_yslope = [](const HalfPlane & hp, Scalar x) -> std::pair<Scalar, Scalar> {
    // ax + by <=> c  for  b != 0   <==>   y <=> c/b - (a/b) x
    const Scalar alpha = -hp.a / hp.b;
    const Scalar beta  = hp.c / hp.b;

    // ensure we can evaluate at \pm inf
    if (std::abs(alpha) <= eps) { return {beta, alpha}; }

    return {alpha * x + beta, alpha};
  };

  /**
   * @brief Compute intersection between two halfplanes
   */
  const std::optional<Scalar> intersection(const HalfPlane & hp1, const HalfPlane & hp2)
  {
    const Scalar lhs = hp1.a * hp2.b - hp2.a * hp1.b;
    if (std::abs(lhs) > eps) { return (hp2.b * hp1.c - hp1.b * hp2.c) / lhs; }
    return {};
  }

  // g(x) = max { ai * x + bi },    and its subderivative
  const auto gfun = [](const std::ranges::range auto & hps, const Scalar x) -> ValSubDer {
    ValSubDer ret{-inf, 0, 0};
    for (const auto & hp : hps | std::views::filter(active_y_lower)) {
      const auto [yx, dyx] = hp_to_yslope(hp, x);
      if (std::get<0>(ret) > yx + eps) {
        // do noting
      } else if (yx > std::get<0>(ret) + eps) {
        ret = {yx, dyx, dyx};
      } else {
        std::get<1>(ret) = std::min(std::get<1>(ret), dyx);
        std::get<2>(ret) = std::max(std::get<2>(ret), dyx);
      }
    }
    return ret;
  };

  // h(x) = min { a * x + b },   and its subderivative
  const auto hfun = [](const std::ranges::range auto & hps, const Scalar x) -> ValSubDer {
    ValSubDer ret{inf, 0, 0};
    for (const auto & hp : hps | std::views::filter(active_y_upper)) {
      const auto [yx, dyx] = hp_to_yslope(hp, x);
      if (std::get<0>(ret) + eps < yx) {
        // do noting
      } else if (yx + eps < std::get<0>(ret)) {
        ret = {yx, dyx, dyx};
      } else {
        std::get<1>(ret) = std::min(std::get<1>(ret), dyx);
        std::get<2>(ret) = std::max(std::get<2>(ret), dyx);
      }
    }
    return ret;
  };

  /// @brief Find candidate optimal point among halfplanes by considering pairwise intersections.
  std::optional<Scalar> find_candidate(std::vector<HalfPlane> & hps, Scalar a, Scalar b)
  {
    std::optional<typename std::vector<HalfPlane>::iterator> it1_store{};

    // keep track of median using two priority queues
    std::priority_queue<Scalar> small, large;
    const auto addnum = [&small, &large](Scalar d) {
      small.push(d);
      large.push(-small.top());
      small.pop();
      while (small.size() < large.size()) {
        small.push(-large.top());
        large.pop();
      }
    };

    // INTERSECT LOWERS AMONGST THEMSELVES

    for (auto it2 = hps.begin(); it2 != hps.end(); ++it2) {

      if (!active_y_lower(*it2)) { continue; }

      if (!it1_store.has_value()) {
        it1_store = it2;
        continue;
      }

      auto it1 = *it1_store;

      const auto isec = intersection(*it1, *it2);

      int8_t redundant = 0;  // 0 (none), 1, or 2

      if (isec.has_value()) {
        if (a + eps < *isec && *isec + eps < b) {
          addnum(*isec);
          it1_store = {};
        } else {                   // intersection outside--one is redundant
          if (a + eps >= *isec) {  // check for redundancy at a
            const auto [v1, dv1] = hp_to_yslope(*it1, a);
            const auto [v2, dv2] = hp_to_yslope(*it2, a);
            if (v1 + eps < v2) {
              redundant = 1;
            } else if (v2 + eps < v1) {
              redundant = 2;
            } else {
              redundant = dv1 <= dv2 ? 1 : 2;
            }
          } else if (*isec + eps >= b) {  // check for redundancy at b
            const auto [v1, dv1] = hp_to_yslope(*it1, b);
            const auto [v2, dv2] = hp_to_yslope(*it2, b);
            if (v1 + eps < v2) {
              redundant = 1;
            } else if (v2 + eps < v1) {
              redundant = 2;
            } else {
              redundant = dv1 >= dv2 ? 1 : 2;
            }
          }
        }
      } else {  // parallel--so one is redundant
        redundant = hp_to_yslope(*it1, 0) < hp_to_yslope(*it2, 0) ? 1 : 2;
      }

      if (redundant == 1) {
        it1->active = false;
        it1_store   = it2;
      } else if (redundant == 2) {
        it2->active = false;
      }
    }

    // INTERSECT UPPERS AMONGST THEMSELVES

    it1_store = {};

    for (auto it2 = hps.begin(); it2 != hps.end(); ++it2) {

      if (!active_y_upper(*it2)) { continue; }

      if (!it1_store.has_value()) {
        it1_store = it2;
        continue;
      }

      auto it1 = *it1_store;

      const auto isec = intersection(*it1, *it2);

      int8_t redundant = 0;  // 0 (none), 1, or 2

      if (isec.has_value()) {
        if (a + eps < *isec && *isec + eps < b) {
          addnum(*isec);
          it1_store = {};
        } else {                   // intersection outside--one is redundant
          if (a + eps >= *isec) {  // check for redundancy at a
            const auto [v1, dv1] = hp_to_yslope(*it1, a);
            const auto [v2, dv2] = hp_to_yslope(*it2, a);
            if (v1 + eps < v2) {
              redundant = 2;
            } else if (v2 + eps < v1) {
              redundant = 1;
            } else {
              redundant = dv1 <= dv2 ? 2 : 1;
            }
          } else if (*isec + eps >= b) {  // check for redundancy at b
            const auto [v1, dv1] = hp_to_yslope(*it1, b);
            const auto [v2, dv2] = hp_to_yslope(*it2, b);
            if (v1 + eps < v2) {
              redundant = 2;
            } else if (v2 + eps < v1) {
              redundant = 1;
            } else {
              redundant = dv1 >= dv2 ? 2 : 1;
            }
          }
        }
      } else {  // parallel--so one is redundant
        redundant = hp_to_yslope(*it1, 0) < hp_to_yslope(*it2, 0) ? 2 : 1;
      }

      if (redundant == 1) {
        it1->active = false;
        it1_store   = it2;
      } else if (redundant == 2) {
        it2->active = false;
      }
    }

    // IF NO POINTS WERE FOUND AND THERE'S A SINGLE LOWER, INTERSECT IT WITH THE UPPERS

    if (small.empty() && std::count_if(hps.cbegin(), hps.cend(), active_y_lower) == 1) {
      const auto hp_l =
        *std::find_if(std::ranges::begin(hps), std::ranges::end(hps), active_y_lower);
      for (auto & hp_u : hps | std::views::filter(active_y_upper)) {
        const auto isec = intersection(hp_l, hp_u);
        if (isec.has_value() && a + eps < *isec && *isec + eps < b) { addnum(*isec); }
      }
    }

    if (small.empty()) { return {}; }

    // return median element
    return small.top();
  }

  /**
   * @brief Check point
   * @param hps half planes defining LP
   * @param x point to check
   * @return
   * - 0 if x is optimal
   * - 1 if optimal solution is to the left of x (if it exists)
   * - 2 if optimal solution is to the right of x (if it exists)
   * - 3 if problem is infeasible
   */
  inline uint8_t check(const std::vector<HalfPlane> & hps, const Scalar x)
  {
    const auto [gx, sg, Sg] = gfun(hps, x);
    const auto [hx, sh, Sh] = hfun(hps, x);

    if (gx <= hx + eps) {   // FEASIBLE
      if (gx + eps < hx) {  // there's slack, only g matters
        if (sg > 0) {
          return 1;
        } else if (Sg < 0) {
          return 2;
        } else {
          return 0;
        }
      } else {  // no slack
        if (sg > 0 && sg >= Sh) {
          return 1;
        } else if (Sg < 0 && Sg < sh) {
          return 2;
        } else {
          return 0;
        }
      }
    } else {  // INFEASIBLE
      if (Sg < sh) {
        return 2;
      } else if (sg > Sh) {
        return 1;
      } else {
        return 3;
      }
    }
  }

  /**
   * @brief Solve 2D linear program
   *
   *  min  y
   *  s.t. a x + by <= c   for (a, b, c) in hps
   *
   * @param hps half plane triplets (a, b, c) defining the LP
   * @return {x, y} optimal solution
   *
   * If problem is infeasible y = inf is returned
   */
  inline std::tuple<Scalar, Scalar, Status> solve_impl(std::vector<HalfPlane> & hps)
  {
    // halfplanes that define a lower bound on x (independent of y)
    auto hps_x_lower = hps | std::views::filter([](const auto & hp) {
                         return hp.active && std::abs(hp.b) < eps && hp.a < 0;
                       });

    // initial lower bound on x
    Scalar a = std::transform_reduce(
      std::ranges::begin(hps_x_lower),
      std::ranges::end(hps_x_lower),
      -inf,
      [](const Scalar a, const Scalar b) { return std::max(a, b); },
      [](const HalfPlane & hp) { return hp.c / hp.a; });

    // halfplanes that define an upper bound on x (independent of y)
    auto hps_x_upper = hps | std::views::filter([](const auto & hp) {
                         return hp.active && std::abs(hp.b) < eps && hp.a > 0;
                       });

    // initial upper bound on x
    Scalar b = std::transform_reduce(
      std::ranges::begin(hps_x_upper),
      std::ranges::end(hps_x_upper),
      inf,
      [](const Scalar a, const Scalar b) { return std::min(a, b); },
      [](const HalfPlane & hp) { return hp.c / hp.a; });

    // we remove at least one halfplane per iterations, so need at most N iterations
    for (auto iter = hps.size(); iter > 0; --iter) {
      const auto x = find_candidate(hps, a, b);

      // remove hps that were marked as not active
      hps.erase(
        std::remove_if(hps.begin(), hps.end(), [](const auto & hp) { return !hp.active; }),
        hps.end());

      if (!x.has_value()) { break; }

      switch (check(hps, *x)) {
      case 0:
        return {*x, std::get<0>(gfun(hps, *x)), Status::Optimal};
        break;
      case 1:
        b = *x;
        break;
      case 2:
        a = *x;
        break;
      case 3:
        return {0, inf, Status::PrimaryInfeasible};
        break;
      }
    }

    // no intersection points, only need to consider boundaries
    const auto [ga, sga, Sga] = gfun(hps, a);
    const auto [ha, sha, Sha] = hfun(hps, a);

    const auto [gb, sgb, Sgb] = gfun(hps, b);
    const auto [hb, shb, Shb] = hfun(hps, b);

    if (ga == ha && gb == hb && std::abs(ga) == inf && std::abs(gb) == inf) {
      // special case where bounds are equal and \pm inf
      if (gfun(hps, 0) <= hfun(hps, 0)) {
        return {0., 0., Status::DualInfeasible};
      } else {
        return {0., 0., Status::PrimaryInfeasible};
      }
    }

    if (ga > ha && gb > hb) {
      return {0, 0, Status::PrimaryInfeasible};
    } else if ((ga <= ha && gb > hb) || ga < gb) {
      return {a, ga, ga == -inf ? Status::DualInfeasible : Status::Optimal};
    } else {
      return {b, gb, gb == -inf ? Status::DualInfeasible : Status::Optimal};
    }
  }

}  // namespace detail

}  // namespace lp2d

#endif  // LP2D__LP2D_HPP_
