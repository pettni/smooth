#ifndef SMOOTH__TRAITS_HPP_
#define SMOOTH__TRAITS_HPP_

#include <cstddef>
#include <utility>


namespace smooth
{

/**
 * @brief Compile-time for loop
 */
template<std::size_t i, std::size_t iend, typename F>
inline static constexpr void static_for_impl(F && f)
{
  if constexpr (i < iend) {
    f(std::integral_constant<std::size_t, i>());
    static_for_impl<i + 1, iend>(std::forward<F>(f));
  }
}

template<std::size_t iend, typename F>
inline static constexpr void static_for(F && f)
{
  static_for_impl<0, iend>(std::forward<F>(f));
}


/**
 * @brief Trait to change templated type
 */
template<typename T, typename ... NewS>
struct change_template_args {};

template<template<typename ...> typename T, typename ... S, typename ... NewS>
struct change_template_args<T<S ...>, NewS ...>
{
  using type = T<NewS ...>;
};

template<typename T, typename ... NewS>
using change_template_args_t = typename change_template_args<T, NewS...>::type;


/**
 * @brief Sum an index sequence
 */
template<typename>
struct iseq_sum {};

template<std::size_t ... _Idx>
struct iseq_sum<std::index_sequence<_Idx...>>
{
  static constexpr std::size_t value = (_Idx + ... + 0);
};


/**
 * @brief Get the element of an index sequence
 */
template<std::size_t, typename>
struct iseq_el {};

template<std::size_t _Beg, std::size_t ... _Idx>
struct iseq_el<0, std::index_sequence<_Beg, _Idx...>>
{
  static constexpr std::size_t value = _Beg;
};
template<std::size_t _I, std::size_t _Beg, std::size_t ... _Idx>
struct iseq_el<_I, std::index_sequence<_Beg, _Idx...>>
  : public iseq_el<_I - 1, std::index_sequence<_Idx...>>
{};

template<std::size_t _I, typename _Seq>
static constexpr std::size_t iseq_el_v = iseq_el<_I, _Seq>::value;


/**
 * @brief Get the length of an index sequence
 */
template<typename>
struct iseq_len {};

template<std::size_t ... _Idx>
struct iseq_len<std::index_sequence<_Idx...>>
{
  static constexpr std::size_t value = sizeof...(_Idx);
};


/**
 * @brief prefix-sum an index sequence
 */
template<typename _Collected, typename _Remaining, std::size_t Sum>
struct iseq_psum_impl;

template<std::size_t... _Cur, std::size_t _Sum>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<>, _Sum>
{
  using type = std::index_sequence<_Cur...>;
};

template<std::size_t _First, std::size_t _Sum, std::size_t... _Cur, std::size_t... _Rem>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<_First, _Rem...>, _Sum>
  : public iseq_psum_impl<std::index_sequence<_Cur..., _Sum>, std::index_sequence<_Rem...>,
    _Sum + _First>
{};

template<typename _Seq>
using iseq_psum = iseq_psum_impl<std::index_sequence<>, _Seq, 0>;


}  // namespace smooth

#endif  // SMOOTH__TRAITS_HPP_
