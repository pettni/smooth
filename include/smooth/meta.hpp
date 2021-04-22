#ifndef SMOOTH__META_HPP_
#define SMOOTH__META_HPP_

#include <cstddef>
#include <utility>


namespace smooth::meta
{

/////////////////////
// STATIC FOR LOOP //
/////////////////////

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


//////////////////////////
// INDEX SEQUENCE UTILS //
//////////////////////////

/**
 * @brief Add constant value X to an integer sequence
 */
template<std::size_t X, typename Iseq>
struct iseq_add;

template<std::size_t X, std::size_t ... Idx>
struct iseq_add<X, std::index_sequence<Idx...>>
{
  using type = std::index_sequence<X + Idx ...>;
};

template<std::size_t X, typename Iseq>
using iseq_add_t = typename iseq_add<X, Iseq>::type;


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

template<typename Iseq>
static constexpr std::size_t iseq_sum_v = iseq_sum<Iseq>::value;


/**
 * @brief Get the N:th element of an index sequence
 */
template<std::size_t N, typename>
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

template<typename Iseq>
static constexpr std::size_t iseq_len_v = iseq_len<Iseq>::value;


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

template<typename _Seq>
using iseq_psum_t = typename iseq_psum_impl<std::index_sequence<>, _Seq, 0>::type;


/////////////////////
// TYPE PACK UTILS //
/////////////////////

/**
 * @brief Type that holds pack of types
 */
template<typename ... Ts>
struct typepack
{
  template<std::size_t Idx>
  using part = std::tuple_element_t<Idx, std::tuple<Ts...>>;

  using tuple = std::tuple<Ts...>;

  template<template<typename ...> typename T>
  using apply = T<Ts...>;

  static constexpr std::size_t size = sizeof...(Ts);
};

/**
 * @brief Select a subset of a typepack using an index sequence
 */
template<typename Iseq, typename Tpack>
struct typepack_select;

template<std::size_t ... Idx, typename ... Ts>
struct typepack_select<std::index_sequence<Idx...>, typepack<Ts...>>
{
  using type = typepack<typename typepack<Ts...>::template part<Idx>...>;
};

template<typename ISeq, typename TPack>
using typepack_select_t = typename typepack_select<ISeq, TPack>::type;

/**
 * @brief Take first N elements of typepack
 */
template<std::size_t N, typename TPack>
struct typepack_take
{
  using type = typepack_select<std::make_index_sequence<N>, TPack>::type;
};

template<std::size_t N, typename TPack>
using typepack_take_t = typename typepack_take<N, TPack>::type;

/**
 * @brief Drop first N elements of typepack
 */
template<std::size_t N, typename TPack>
struct typepack_drop
{
  using type = typepack_select_t<iseq_add_t<N, std::make_index_sequence<TPack::size - N>>, TPack>;
};

template<std::size_t N, typename TPack>
using typepack_drop_t = typename typepack_drop<N, TPack>::type;

/**
 * @brief Concatenate multiple typepacks
 */
template<typename ... TPacks>
struct typepack_cat;

template<typename TPack>
struct typepack_cat<TPack>
{
  using type = TPack;
};

template<typename ... Ts1, typename ... Ts2>
struct typepack_cat<typepack<Ts1...>, typepack<Ts2...>>
{
  using type = typepack<Ts1..., Ts2...>;
};

template<typename TPack1, typename TPack2, typename ... TPacks>
struct typepack_cat<TPack1, TPack2, TPacks...>
{
  using type = typepack_cat<typename typepack_cat<TPack1, TPack2>::type, TPacks...>::type;
};

template<typename ... TPacks>
using typepack_cat_t = typename typepack_cat<TPacks...>::type;


///////////////////////////
// TYPE META PROGRAMMING //
///////////////////////////

/**
 * @brief Change the N:th template argument in T<Ts...> to NewT
 *
 * @tparam T
 * @tparam Idx
 * @tparam New
 */
template<typename T, std::size_t N, typename NewT>
struct change_template_arg;

template<template<typename ...> typename T, std::size_t N, typename New, typename ... Ts>
struct change_template_arg<T<Ts...>, N, New>
{
  using type = typename
    typepack_cat_t<
    typepack_take_t<N, typepack<Ts...>>,
    typepack<New>,
    typepack_drop_t<N + 1, typepack<Ts...>>
    >::apply<T>;
};

template<typename T, std::size_t N, typename NewT>
using change_template_arg_t = typename change_template_arg<T, N, NewT>::type;


/////////////////////////////////
// COMPILE-TIME MATRIX ALGEBRA //
/////////////////////////////////

/**
 * @brief Elementary structure for compile-time matrix algebra
 */
template<typename Scalar, std::size_t Rows, std::size_t Cols>
struct StaticMatrix : std::array<std::array<Scalar, Cols>, Rows>
{
  using std::array<std::array<Scalar, Cols>, Rows>::operator[];

  /**
   * @brief Construct a matrix filled with zeros
   */
  constexpr StaticMatrix()
  : std::array<std::array<Scalar, Cols>, Rows>{}
  {
    for (auto i = 0u; i != Rows; ++i) {
      operator[](i).fill(Scalar(0));
    }
  }

  /**
   * @brief Add two matrices
   */
  constexpr StaticMatrix<Scalar, Rows, Cols> operator+(StaticMatrix<Scalar, Rows, Cols> o) const
  {
    StaticMatrix<Scalar, Rows, Cols> ret;
    for (auto i = 0u; i < Rows; ++i) {
      for (auto j = 0u; j < Cols; ++j) {
        ret[i][j] = operator[](i)[j] + o[i][j];
      }
    }
    return ret;
  }

  /**
   * @brief Return transpose of a matrix
   */
  constexpr StaticMatrix<Scalar, Rows, Cols> transpose() const
  {
    StaticMatrix<Scalar, Rows, Cols> ret;
    for (auto i = 0u; i < Rows; ++i) {
      for (auto j = 0u; j < Cols; ++j) {
        ret[j][i] = operator[](i)[j];
      }
    }
    return ret;
  }

  /**
   * @brief Multiply two matrices
   */
  template<std::size_t ColsNew>
  constexpr StaticMatrix<Scalar, Rows, ColsNew>
  operator*(StaticMatrix<Scalar, Cols, ColsNew> o) const
  {
    StaticMatrix<Scalar, Rows, ColsNew> ret;
    for (auto i = 0u; i < Rows; ++i) {
      for (auto j = 0u; j < ColsNew; ++j) {
        for (auto k = 0u; k < Cols; ++k) {
          ret[i][j] += operator[](i)[k] * o[k][j];
        }
      }
    }
    return ret;
  }
};

}  // namespace smooth::meta

#endif  // SMOOTH__META_HPP_
