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

}  // namespace smooth

#endif  // SMOOTH__TRAITS_HPP_
