// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ranges>
#include <tuple>
#include <utility>

#include "smooth/version.hpp"

SMOOTH_BEGIN_NAMESPACE

namespace utils {
////////////////////////////
// INTERVAL BINARY SEARCH //
////////////////////////////

/**
 * @brief Find interval in sorted range with binary search.
 *
 * 1. If r is empty, returns r.end()                  (not found)
 * 2. If t < r.front(), returns r.end()               (not found)
 * 3. If t >= r.back(), returns r.end() - 1           (no upper bound)
 * 4. Otherwise returns it s.t. *it <= t < *(it + 1)
 *
 * @param r sorted range to search in
 * @param t value to search for
 * @param wo comparison operation with signature \p std::weak_ordering(const
 * std::ranges::range_value_t<_R> &, const _T &)
 *
 * @return range iterator it according to the above rules
 */
constexpr auto binary_interval_search(std::ranges::random_access_range auto && r, auto && t, auto && wo) noexcept
{
  using T  = std::decay_t<decltype(t)>;
  using Rv = std::ranges::range_value_t<std::decay_t<decltype(r)>>;

  auto left = std::ranges::cbegin(r);
  auto rght = std::ranges::cend(r);

  if (std::ranges::empty(r) || wo(*left, t) > 0) {
    return rght;
  } else if (wo(*(rght - 1), t) <= 0) {
    return rght - 1;
  }

  auto pivot = left;

  while (left + 1 < rght) {
    double alpha = 0.5;
    if constexpr (std::is_convertible_v<Rv, double> && std::is_convertible_v<T, double>) {
      alpha = (static_cast<double>(t) - static_cast<double>(*left)) / static_cast<double>(*(rght - 1) - *left);
    }
    const auto dist = static_cast<double>(std::distance(left, rght - 1));
    pivot           = std::ranges::next(left, static_cast<std::intptr_t>(alpha * dist), rght - 2);

    if (wo(*next(pivot), t) <= 0) {
      left = pivot + 1;
    } else if (wo(*pivot, t) > 0) {
      rght = pivot + 1;
    } else {
      break;
    }
  }

  return pivot;
}

/**
 * @brief Find interval in sorted range with binary search using default comparison.
 */
constexpr auto binary_interval_search(std::ranges::random_access_range auto && r, auto && t) noexcept
{
  return binary_interval_search(
    std::forward<decltype(r)>(r), std::forward<decltype(t)>(t), [](const auto & _s, const auto & _t) {
      return _s <=> _t;
    });
}

/////////////////////
// STATIC FOR LOOP //
/////////////////////

/**
 * @brief Compile-time for loop equivalent to the statement (f(0), f(1), ..., f(_I-1))
 */
template<std::size_t _I>
constexpr auto static_for(auto && f) noexcept(noexcept(std::invoke(f, std::integral_constant<std::size_t, 0>())))
{
  const auto f_caller = [&]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
    return (std::invoke(f, std::integral_constant<std::size_t, _Idx>()), ...);
  };

  return f_caller(std::make_index_sequence<_I>{});
}

/////////////////
// ARRAY UTILS //
/////////////////

/**
 * @brief Prefix-sum an array starting at zero
 */
template<typename _T, std::size_t _L>
constexpr std::array<_T, _L + 1> array_psum(const std::array<_T, _L> & x) noexcept
{
  std::array<_T, _L + 1> ret;
  ret[0] = _T(0);
  std::partial_sum(x.begin(), x.end(), std::next(ret.begin(), 1));
  return ret;
}

/////////////////
// RANGE UTILS //
/////////////////

// \cond

// Pairwise transform views

/// @brief Apply function to pairwise elements
template<std::ranges::input_range R, std::copy_constructible F>
  requires std::ranges::view<R>
class pairwise_transform_view : public std::ranges::view_interface<pairwise_transform_view<R, F>>
{
public:
  class _Iterator
  {
  private:
    const pairwise_transform_view * parent_;
    std::ranges::iterator_t<const R> it1_, it2_;

  public:
    using value_type = std::remove_cvref_t<
      std::invoke_result_t<F &, std::ranges::range_reference_t<R>, std::ranges::range_reference_t<R>>>;
    using difference_type = std::ranges::range_difference_t<R>;

    _Iterator() = default;

    constexpr _Iterator(const pairwise_transform_view * parent, const R & r)
        : parent_(parent), it1_(std::ranges::begin(r)), it2_(std::ranges::begin(r))
    {
      if (it2_ != std::ranges::end(r)) { ++it2_; }
    }

    constexpr decltype(auto) operator*() const { return std::invoke(parent_->f_, *it1_, *it2_); }

    constexpr _Iterator & operator++()
    {
      ++it1_, ++it2_;
      return *this;
    }

    constexpr void operator++(int) { ++it1_, ++it2_; }

    constexpr _Iterator operator++(int)
      requires std::ranges::forward_range<R>
    {
      _Iterator tmp = *this;
      ++this;
      return tmp;
    }

    constexpr _Iterator & operator--()
      requires std::ranges::bidirectional_range<R>
    {
      --it1_, --it2_;
      return *this;
    }

    constexpr _Iterator operator--(int)
      requires std::ranges::bidirectional_range<R>
    {
      auto tmp = *this;
      --this;
      return tmp;
    }

    friend constexpr bool operator==(const _Iterator & x, const _Iterator & y) { return x.it1_ == y.it1_; }

    friend constexpr bool operator==(const _Iterator & x, const std::ranges::sentinel_t<const R> & y)
    {
      return x.it2_ == y;
    }
  };

private:
  R base_{};
  F f_{};

public:
  template<typename Fp>
  constexpr pairwise_transform_view(R base, Fp && f) : base_(base), f_(std::forward<Fp>(f))
  {}

  constexpr _Iterator begin() const { return _Iterator(this, base_); }

  constexpr std::ranges::sentinel_t<const R> end() const { return std::ranges::end(base_); }

  constexpr auto size() const
    requires std::ranges::sized_range<const R>
  {
    const auto s = std::ranges::size(base_);
    return (s >= 2) ? s - 1 : 0;
  }
};

/// @brief Deduction guide for pairwise_transform_view
template<typename R, typename F>
pairwise_transform_view(R &&, F) -> pairwise_transform_view<std::views::all_t<R>, F>;

namespace detail {

template<typename F>
struct PairwiseTransformClosure
{
  F f_;

  explicit constexpr PairwiseTransformClosure(F && f) : f_(std::forward<F>(f)) {}

  template<std::ranges::viewable_range R>
  constexpr auto operator()(R && r) const
  {
    return pairwise_transform_view(std::forward<R>(r), f_);
  }
};

struct PairwiseTransform
{
  template<std::ranges::viewable_range R, typename F>
  constexpr auto operator()(R && r, F && f) const
  {
    return pairwise_transform_view(std::forward<R>(r), std::forward<F>(f));
  }

  template<typename F>
  constexpr auto operator()(F && f) const
  {
    return PairwiseTransformClosure<F>(std::forward<F>(f));
  }
};

template<std::ranges::viewable_range R, typename F>
constexpr auto operator|(R && r, const PairwiseTransformClosure<F> & closure)
{
  return closure(std::forward<R>(r));
}

}  // namespace detail

// \endcond

namespace views {

/// @brief Apply function to pairwise elements
inline constexpr detail::PairwiseTransform pairwise_transform;

}  // namespace views

// \cond

// Zip view

/// @brief Zip views
template<std::ranges::input_range... View>
  requires(std::ranges::view<View> && ...)
class zip_view : public std::ranges::view_interface<zip_view<View...>>
{
public:
  template<bool Const>
  class _Iterator
  {
  private:
    std::tuple<std::ranges::iterator_t<std::conditional_t<Const, const View, View>>...> its_;

  public:
    using value_type      = std::tuple<std::ranges::range_value_t<View>...>;
    using difference_type = std::common_type_t<std::ranges::range_difference_t<View>...>;

    _Iterator() = default;

    explicit constexpr _Iterator(auto &&... its) : its_(its...) {}

    explicit constexpr _Iterator(_Iterator<!Const> i)
      requires Const
        : its_(i.its_)
    {}

    constexpr decltype(auto) operator*() const
    {
      return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tuple<std::ranges::range_reference_t<View>...>(std::get<Idx>(its_).operator*()...);
      }(std::make_index_sequence<sizeof...(View)>{});
    }

    constexpr _Iterator & operator++()
    {
      [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        (++std::get<Idx>(its_), ...);
      }(std::make_index_sequence<sizeof...(View)>{});
      return *this;
    }

    constexpr void operator++(int)
    {
      [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        (++std::get<Idx>(its_), ...);
      }(std::make_index_sequence<sizeof...(View)>{});
    }

    constexpr _Iterator operator++(int)
      requires(std::ranges::forward_range<View> && ...)
    {
      _Iterator tmp = *this;
      ++this;
      return tmp;
    }

    friend constexpr bool operator==(const _Iterator & x, const _Iterator & y) { return x.its_ == y.its_; }

    friend constexpr bool operator==(
      const _Iterator & x,
      const std::tuple<std::ranges::sentinel_t<std::conditional_t<Const, const View, View>>...> & y)
    {
      return [&x, &y]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return ((std::get<Idx>(x.its_) == std::get<Idx>(y)) || ...);
      }(std::make_index_sequence<sizeof...(View)>{});
    }
  };  // _Iterator

private:
  std::tuple<View...> bases_{};

public:
  constexpr zip_view() = default;

  explicit constexpr zip_view(View... base) : bases_(base...) {}

  constexpr _Iterator<false> begin()
  {
    return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return _Iterator<false>(std::ranges::begin(std::get<Idx>(bases_))...);
    }(std::make_index_sequence<sizeof...(View)>{});
  }

  // some views only support mutable iteration (e.g. drop view over non-random access view). zip can
  // only allow const iteration if all underlying views support it
  constexpr _Iterator<true> begin() const
    requires(std::ranges::range<const View> && ...)
  {
    return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return _Iterator<true>(std::ranges::begin(std::get<Idx>(bases_))...);
    }(std::make_index_sequence<sizeof...(View)>{});
  }

  constexpr decltype(auto) end()
  {
    return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return std::make_tuple<std::ranges::sentinel_t<View>...>(std::ranges::end(std::get<Idx>(bases_))...);
    }(std::make_index_sequence<sizeof...(View)>{});
  }

  // see comment for const begin
  constexpr decltype(auto) end() const
    requires(std::ranges::range<const View> && ...)
  {
    return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return std::make_tuple<std::ranges::sentinel_t<const View>...>(std::ranges::end(std::get<Idx>(bases_))...);
    }(std::make_index_sequence<sizeof...(View)>{});
  }

  constexpr auto size() const
    requires(std::ranges::sized_range<View> && ...)
  {
    return [this]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return std::min({std::ranges::size(std::get<Idx>(bases_))...});
    }(std::make_index_sequence<sizeof...(View)>{});
  }
};

/// @brief Deduction guide for zip_view
template<std::ranges::viewable_range... R>
zip_view(R &&...) -> zip_view<std::views::all_t<R>...>;

namespace detail {

struct Zip
{
  template<std::ranges::viewable_range... R>
  constexpr auto operator()(R &&... r) const
  {
    return zip_view(std::forward<R>(r)...);
  }
};

}  // namespace detail

// \endcond

/// @brief Zip views
inline constexpr detail::Zip zip;

}  // namespace utils

SMOOTH_END_NAMESPACE
