#include "smooth/concepts.hpp"

namespace smooth
{

namespace detail
{

/**
 * @brief Elementary structure for compile-time matrix algebra
 */
template<typename Scalar, std::size_t Rows, std::size_t Cols>
struct StaticMatrix : std::array<std::array<Scalar, Cols>, Rows>
{
  using std::array<std::array<Scalar, Cols>, Rows>::operator[];

  constexpr StaticMatrix()
  : std::array<std::array<Scalar, Cols>, Rows>{}
  {
    for (auto i = 0u; i != Rows; ++i) {
      operator[](i).fill(Scalar(0));
    }
  }

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

template<typename Scalar, std::size_t K>
constexpr StaticMatrix<Scalar, K + 1, K + 1> card_coeffmat()
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;
  if constexpr (K == 0) {
    ret[0][0] = 1;
    return ret;
  } else {
    constexpr auto coeff_mat_km1 = card_coeffmat<Scalar, K - 1>();
    StaticMatrix<Scalar, K + 1, K> low, high;
    StaticMatrix<Scalar, K, K + 1> left, right;

    for (std::size_t i = 0; i != K; ++i) {
      for (std::size_t j = 0; j != K; ++j) {
        low[i][j] = coeff_mat_km1[i][j];
        high[i + 1][j] = coeff_mat_km1[i][j];
      }
    }

    for (std::size_t k = 0; k != K; ++k) {
      left[k][k + 1] = static_cast<Scalar>(K - (k + 1)) / static_cast<Scalar>(K);
      left[k][k] = Scalar(1) - left[k][k + 1];

      right[k][k + 1] = Scalar(1) / static_cast<Scalar>(K);
      right[k][k] = -right[k][k + 1];
    }

    return low * left + high * right;
  }
}

template<typename Scalar, std::size_t K>
constexpr StaticMatrix<Scalar, K + 1, K + 1> cum_card_coeffmat()
{
  auto ret = card_coeffmat<Scalar, K>();
  for (std::size_t i = 0; i != K + 1; ++i) {
    for (std::size_t j = 0; j != K; ++j) {
      ret[i][K - 1 - j] += ret[i][K - j];
    }
  }
  return ret;
}

}  // namespace detail


/**
 * @brief Evaluate a cardinal bspline of order K
 *
 * interval [beg, end] must be of size K+1
 *
 * @tparam G lie group type
 * @tparam K bspline order
 * @tparam It iterator type
 * @param beg start of control points
 * @param end end of control points
 * @param u interval
 */
template<LieGroupLike G, std::size_t K = 3, typename It>
G bspline(
  It begin, It end, typename G::Scalar u,
  std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
  std::optional<Eigen::Ref<typename G::Tangent>> acc = {}
)
{
  using Scalar = typename G::Scalar;
  Eigen::Matrix<Scalar, 1, K + 1> uvec, duvec, d2uvec;

  if (end - begin != K + 1) {
    throw std::runtime_error("bspline: wrong number of control points");
  }

  uvec(0) = Scalar(1);
  for (std::size_t k = 1; k != K + 1; ++k) {
    uvec(k) = u * uvec(k - 1);
    if (vel.has_value() || acc.has_value()) {
      duvec(k) = Scalar(k) * uvec(k - 1);
      if (acc.has_value()) {
        d2uvec(k) = Scalar(k - 1) * duvec(k - 1);
      }
    }
  }

  // transpose to read rows that are consecutive in memory
  constexpr auto Ms = detail::cum_card_coeffmat<double, K>().transpose();

  Eigen::Matrix<Scalar, K + 1, K + 1> M =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1>>(Ms[0].data()).template cast<Scalar>();

  G g = *begin++;
  if (vel.has_value() || acc.has_value()) {
    vel.value().setZero();
    if (acc.has_value()) {
      acc.value().setZero();
    }
  }

  for (std::size_t j = 1; j != K + 1; ++j) {
    const Scalar Btilde = uvec.dot(M.row(j));
    const typename G::Tangent v = ((*(begin - 1)).inverse() * *begin).log();
    g *= G::exp(Btilde * v);

    if (vel.has_value() || acc.has_value()) {
      const Scalar dBtilde = duvec.dot(M.row(j));
      const auto Ad = G::exp(-Btilde * v).Ad();
      vel.value().applyOnTheLeft(Ad);
      vel.value() += dBtilde * v;

      if (acc.has_value()) {
        const Scalar d2Btilde = d2uvec.dot(M.row(j));
        acc.value().applyOnTheLeft(Ad);
        acc.value() += dBtilde * G::ad(vel.value()) * v + d2Btilde * v;
      }
    }
    ++begin;
  }

  return g;
}

} // namespace smooth
