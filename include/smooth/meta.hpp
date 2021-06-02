#ifndef SMOOTH__META_HPP_
#define SMOOTH__META_HPP_

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>


namespace smooth::meta
{

/////////////////////
// STATIC FOR LOOP //
/////////////////////

/**
 * @brief Compile-time for loop implementation
 */
template<typename _F, std::size_t ... _Idx>
inline static constexpr void static_for_impl(_F && f, std::index_sequence<_Idx...>)
{
  (std::invoke(f, std::integral_constant<std::size_t, _Idx>()), ...);
}

/**
 * @brief Compile-time for loop over 0, ..., _I-1
 */
template<std::size_t _I, typename _F>
inline static constexpr void static_for(_F && f)
{
  static_for_impl(std::forward<_F>(f), std::make_index_sequence<_I>{});
}


/////////////////
// ARRAY UTILS //
/////////////////

/**
 * @brief Prefix-sum an array starting at zero
 */
template<typename T, std::size_t L>
constexpr std::array<T, L+1> array_psum(const std::array<T, L> & x)
{
  std::array<T, L+1> ret;
  ret[0] = T(0);
  std::partial_sum(x.begin(), x.end(), ret.begin() + 1);
  return ret;
}


//////////////////////////
// INDEX SEQUENCE UTILS //
//////////////////////////

/**
 * @brief Add constant value X to an integer sequence
 */
template<std::size_t _X, typename _Seq>
struct iseq_add;

template<std::size_t _X, std::size_t ... Idx>
struct iseq_add<_X, std::index_sequence<Idx...>>
{
  using type = std::index_sequence<_X + Idx ...>;
};

template<std::size_t _X, typename _Seq>
using iseq_add_t = typename iseq_add<_X, _Seq>::type;

/**
 * @brief Sum an index sequence
 */
template<typename _Seq>
struct iseq_sum {};

template<std::size_t ... _Is>
struct iseq_sum<std::index_sequence<_Is...>>
{
  static constexpr std::size_t value = (_Is + ... + 0);
};

template<typename _Seq>
static constexpr std::size_t iseq_sum_v = iseq_sum<_Seq>::value;

/**
 * @brief Get the N:th element of an index sequence
 */
template<std::size_t _N, typename _Seq>
struct iseq_el {};

template<std::size_t _Beg, std::size_t ... _Is>
struct iseq_el<0, std::index_sequence<_Beg, _Is...>>
{
  static constexpr std::size_t value = _Beg;
};
template<std::size_t _N, std::size_t _Beg, std::size_t ... _Is>
struct iseq_el<_N, std::index_sequence<_Beg, _Is...>>
  : public iseq_el<_N - 1, std::index_sequence<_Is...>>
{};

template<std::size_t _N, typename _Seq>
static constexpr std::size_t iseq_el_v = iseq_el<_N, _Seq>::value;

/**
 * @brief Get the length of an index sequence
 */
template<typename _Seq>
struct iseq_len {};

template<std::size_t ... _Is>
struct iseq_len<std::index_sequence<_Is...>>
{
  static constexpr std::size_t value = sizeof...(_Is);
};

template<typename _Seq>
static constexpr std::size_t iseq_len_v = iseq_len<_Seq>::value;

/**
 * @brief prefix-sum an index sequence
 */
template<typename _Collected, typename _Remaining, std::size_t _Sum>
struct iseq_psum_impl;

template<std::size_t... _Cur, std::size_t _Sum>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<>, _Sum>
{
  using type = std::index_sequence<_Cur...>;
};

template<std::size_t _First, std::size_t _Sum, std::size_t... _Cur, std::size_t... _Rem>
struct iseq_psum_impl<std::index_sequence<_Cur...>, std::index_sequence<_First, _Rem...>, _Sum>
  : public iseq_psum_impl<std::index_sequence<_Cur..., _Sum>, std::index_sequence<_Rem...>, _Sum + _First>
{};

template<typename _Seq>
using iseq_psum = iseq_psum_impl<std::index_sequence<>, _Seq, 0>;

template<typename _Seq>
using iseq_psum_t = typename iseq_psum_impl<std::index_sequence<>, _Seq, 0>::type;


/////////////////////
// TYPE PACK UTILS //
/////////////////////

/**
 * @brief Type that holds pack of types
 */
template<typename ... _Ts>
struct typepack
{
  template<std::size_t _I>
  using part = std::tuple_element_t<_I, std::tuple<_Ts...>>;

  using tuple = std::tuple<_Ts...>;

  template<template<typename ...> typename T>
  using apply = T<_Ts...>;

  static constexpr std::size_t size = sizeof...(_Ts);
};

/**
 * @brief Select a subset of a typepack using an index sequence
 */
template<typename _Seq, typename _TPack>
struct typepack_select;

template<std::size_t ... _I, typename ... _Ts>
struct typepack_select<std::index_sequence<_I...>, typepack<_Ts...>>
{
  using type = typepack<typename typepack<_Ts...>::template part<_I>...>;
};

template<typename _Seq, typename _TPack>
using typepack_select_t = typename typepack_select<_Seq, _TPack>::type;

/**
 * @brief Take first N elements of typepack
 */
template<std::size_t _N, typename _TPack>
struct typepack_take
{
  using type = typename typepack_select<std::make_index_sequence<_N>, _TPack>::type;
};

template<std::size_t _N, typename _TPack>
using typepack_take_t = typename typepack_take<_N, _TPack>::type;

/**
 * @brief Drop first _N elements of typepack
 */
template<std::size_t _N, typename _TPack>
struct typepack_drop
{
  using type = typepack_select_t<iseq_add_t<_N, std::make_index_sequence<_TPack::size - _N>>, _TPack>;
};

template<std::size_t _N, typename _TPack>
using typepack_drop_t = typename typepack_drop<_N, _TPack>::type;

/**
 * @brief Concatenate multiple typepacks
 */
template<typename ... _TPacks>
struct typepack_cat;

template<typename _TPack>
struct typepack_cat<_TPack>
{
  using type = _TPack;
};

template<typename ... Ts1, typename ... Ts2>
struct typepack_cat<typepack<Ts1...>, typepack<Ts2...>>
{
  using type = typepack<Ts1..., Ts2...>;
};

template<typename _TPack1, typename _TPack2, typename ... _TPacks>
struct typepack_cat<_TPack1, _TPack2, _TPacks...>
{
  using type = typename typepack_cat<typename typepack_cat<_TPack1, _TPack2>::type, _TPacks...>::type;
};

template<typename ... _TPacks>
using typepack_cat_t = typename typepack_cat<_TPacks...>::type;


///////////////////////////
// TYPE META PROGRAMMING //
///////////////////////////

/**
 * @brief Change the _N:th template argument in T<Ts...> to NewT
 */
template<typename _T, std::size_t _N, typename _NewT>
struct change_template_arg;

template<template<typename ...> typename _T, std::size_t _N, typename _NewT, typename ... _Ts>
struct change_template_arg<_T<_Ts...>, _N, _NewT>
{
  using type = typename
    typepack_cat_t<
    typepack_take_t<_N, typepack<_Ts...>>,
    typepack<_NewT>,
    typepack_drop_t<_N + 1, typepack<_Ts...>>
    >::template apply<_T>;
};

template<typename _T, std::size_t _N, typename _NewT>
using change_template_arg_t = typename change_template_arg<_T, _N, _NewT>::type;


/////////////////////////////////
// COMPILE-TIME MATRIX ALGEBRA //
/////////////////////////////////

/**
 * @brief Elementary structure for compile-time matrix algebra
 */
template<typename _Scalar, std::size_t _Rows, std::size_t _Cols>
struct StaticMatrix : public std::array<std::array<_Scalar, _Cols>, _Rows>
{
  std::size_t Rows = _Rows;
  std::size_t Cols = _Cols;

  using std::array<std::array<_Scalar, _Cols>, _Rows>::operator[];

  /**
   * @brief Construct a matrix filled with zeros
   */
  constexpr StaticMatrix()
  : std::array<std::array<_Scalar, _Cols>, _Rows>{}
  {
    for (auto i = 0u; i != _Rows; ++i) {
      operator[](i).fill(_Scalar(0));
    }
  }

  /**
   * @brief Add two matrices
   */
  constexpr StaticMatrix<_Scalar, _Rows, _Cols> operator+(StaticMatrix<_Scalar, _Rows, _Cols> o) const
  {
    StaticMatrix<_Scalar, _Rows, _Cols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _Cols; ++j) {
        ret[i][j] = operator[](i)[j] + o[i][j];
      }
    }
    return ret;
  }

  /**
   * @brief Return transpose of a matrix
   */
  constexpr StaticMatrix<_Scalar, _Rows, _Cols> transpose() const
  {
    StaticMatrix<_Scalar, _Rows, _Cols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _Cols; ++j) {
        ret[j][i] = operator[](i)[j];
      }
    }
    return ret;
  }

  /**
   * @brief Multiply two matrices
   */
  template<std::size_t _ColsNew>
  constexpr StaticMatrix<_Scalar, _Rows, _ColsNew>
  operator*(StaticMatrix<_Scalar, _Cols, _ColsNew> o) const
  {
    StaticMatrix<_Scalar, _Rows, _ColsNew> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _ColsNew; ++j) {
        for (auto k = 0u; k < _Cols; ++k) {
          ret[i][j] += operator[](i)[k] * o[k][j];
        }
      }
    }
    return ret;
  }
};

}  // namespace smooth::meta

#endif  // SMOOTH__META_HPP_
