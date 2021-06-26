#ifndef SMOOTH__UTILS_HPP_
#define SMOOTH__UTILS_HPP_

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>

#include <Eigen/Core>

namespace smooth::utils {

/////////////////////
// STATIC FOR LOOP //
/////////////////////

/**
 * @brief Compile-time for loop implementation
 */
template<typename _F, std::size_t... _Idx>
inline static constexpr auto static_for_impl(_F && f, std::index_sequence<_Idx...>)
{
  return (std::invoke(f, std::integral_constant<std::size_t, _Idx>()), ...);
}

/**
 * @brief Compile-time for loop over 0, ..., _I-1
 */
template<std::size_t _I, typename _F>
inline static constexpr auto static_for(_F && f)
{
  return static_for_impl(std::forward<_F>(f), std::make_index_sequence<_I>{});
}

/////////////////
// ARRAY UTILS //
/////////////////

/**
 * @brief Prefix-sum an array starting at zero
 */
template<typename T, std::size_t L>
constexpr std::array<T, L + 1> array_psum(const std::array<T, L> & x)
{
  std::array<T, L + 1> ret;
  ret[0] = T(0);
  std::partial_sum(x.begin(), x.end(), ret.begin() + 1);
  return ret;
}

/////////////////////
// SUMMATION UTILS //
/////////////////////

template<typename Tuple>
struct tuple_dof
{};

template<typename... Wrt>
struct tuple_dof<std::tuple<Wrt...>>
{
  static constexpr Eigen::Index value =
    std::min<Eigen::Index>({std::decay_t<Wrt>::SizeAtCompileTime...}) == -1
      ? -1
      : (std::decay_t<Wrt>::SizeAtCompileTime + ...);
};

template<typename Scalar, typename... _Wrt, std::size_t... _Idx>
auto tuple_cast(const std::tuple<_Wrt...> & wrt, std::index_sequence<_Idx...>)
{
  using RetType = std::tuple<
    typename std::decay_t<decltype(std::decay_t<_Wrt>{}.template cast<Scalar>())>::PlainObject...>;
  return RetType(std::get<_Idx>(wrt).template cast<Scalar>()...);
}

template<typename Scalar, typename... _Wrt>
auto tuple_cast(const std::tuple<_Wrt...> & wrt)
{
  return tuple_cast<Scalar>(wrt, std::make_index_sequence<sizeof...(_Wrt)>{});
}

/**
 * @brief Add an eigen tangent vector to a tuple of variables
 *
 * \todo Do this in a single function without index seq
 */
template<typename Derived, typename... _Wrt, std::size_t... Idx>
auto tuple_plus(const std::tuple<_Wrt...> & wrt,
  const Eigen::MatrixBase<Derived> & a,
  std::index_sequence<Idx...>)
{
  const std::array<Eigen::Index, sizeof...(_Wrt)> sizes{std::get<Idx>(wrt).size()...};
  const auto sizes_psum = array_psum(sizes);

  return std::tuple<std::decay_t<_Wrt>...>(
    std::get<Idx>(wrt)
    + a.template segment<std::decay_t<_Wrt>::SizeAtCompileTime>(sizes_psum[Idx], sizes[Idx])...);
}

/**
 * @brief Add an eigen tangent vector to a tuple of variables
 */
template<typename Derived, typename... _Wrt>
auto tuple_plus(const std::tuple<_Wrt...> & wrt, const Eigen::MatrixBase<Derived> & a)
{
  return tuple_plus(wrt, a, std::make_index_sequence<sizeof...(_Wrt)>{});
}

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
  constexpr StaticMatrix() : std::array<std::array<_Scalar, _Cols>, _Rows>{}
  {
    for (auto i = 0u; i != _Rows; ++i) { operator[](i).fill(_Scalar(0)); }
  }

  /**
   * @brief Add two matrices
   */
  constexpr StaticMatrix<_Scalar, _Rows, _Cols> operator+(
    StaticMatrix<_Scalar, _Rows, _Cols> o) const
  {
    StaticMatrix<_Scalar, _Rows, _Cols> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _Cols; ++j) { ret[i][j] = operator[](i)[j] + o[i][j]; }
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
      for (auto j = 0u; j < _Cols; ++j) { ret[j][i] = operator[](i)[j]; }
    }
    return ret;
  }

  /**
   * @brief Multiply two matrices
   */
  template<std::size_t _ColsNew>
  constexpr StaticMatrix<_Scalar, _Rows, _ColsNew> operator*(
    StaticMatrix<_Scalar, _Cols, _ColsNew> o) const
  {
    StaticMatrix<_Scalar, _Rows, _ColsNew> ret;
    for (auto i = 0u; i < _Rows; ++i) {
      for (auto j = 0u; j < _ColsNew; ++j) {
        for (auto k = 0u; k < _Cols; ++k) { ret[i][j] += operator[](i)[k] * o[k][j]; }
      }
    }
    return ret;
  }
};

}  // namespace smooth::utils

#endif  // SMOOTH__UTILS_HPP_
