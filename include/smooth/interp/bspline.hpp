#include <iostream>
#include <ranges>

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
 * @brief Return knot indices of the control points
 *
 * Return value represents range i_{-K}, ..., i_1 where i_0 is the
 * left point of the spline interval where t falls.
 *
 * Negative knot indices correspond to exterior knots, the smallest
 * possible knot index is -K
 *
 * @param t time where spline is to be evaluated
 * @param t0 start time of spline
 * @param dt spline knot distance
 * @return std::pair<std::size_t, std::size_t>
 */
template<std::size_t K>
std::pair<int64_t, int64_t> bspline_range(double t, double t0, double dt)
{
  auto idx_ival = static_cast<int64_t>((t - t0) / dt);

  return std::make_pair(idx_ival - K, idx_ival + 1);
}


/**
 * @brief Evaluate a cardinal bspline of order K and calculate derivatives
 *
 *   g = g_0 * \Prod_{i=1}^{K} exp ( Btilde_i(u) * log( g_{i-1}^{-1}  * g_i ) )
 *
 * Where Btilde are cumulative Bspline basis functins.
 *
 * @tparam G lie group type
 * @tparam K bspline order
 * @tparam It iterator type
 * @param[in] ctrl_points range of control points (must be of size K + 1)
 * @param[in] u interval location: u = (t - ti) / dt \in [0, 1)
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 */
template<LieGroupLike G, std::size_t K = 3, std::ranges::range Range>
G bspline_eval(
  Range ctrl_points,
  typename G::Scalar u,
  std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
  std::optional<Eigen::Ref<typename G::Tangent>> acc = {}
)
{
  using Scalar = typename G::Scalar;
  Eigen::Matrix<Scalar, 1, K + 1> uvec, duvec, d2uvec;

  if (std::ranges::size(ctrl_points) != K + 1) {
    throw std::runtime_error(
            "bspline: control point range must be size K+1=" + std::to_string(
              K + 1) + ", got " + std::to_string(std::ranges::size(ctrl_points)));
  }

  uvec(0) = Scalar(1);
  duvec(0) = Scalar(0);
  d2uvec(0) = Scalar(0);

  for (std::size_t k = 1; k != K + 1; ++k) {
    uvec(k) = u * uvec(k - 1);
    if (vel.has_value() || acc.has_value()) {
      duvec(k) = Scalar(k) * uvec(k - 1);
      if (acc.has_value()) {
        d2uvec(k) = Scalar(k) * duvec(k - 1);
      }
    }
  }

  // transpose to read rows that are consecutive in memory
  constexpr auto Ms = detail::cum_card_coeffmat<double, K>().transpose();

  Eigen::Matrix<Scalar, K + 1, K + 1> M =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(Ms[0].data()).template cast<Scalar>();

  if (vel.has_value() || acc.has_value()) {
    vel.value().setZero();
    if (acc.has_value()) {
      acc.value().setZero();
    }
  }

  G g = *std::ranges::begin(ctrl_points);
  G gjm1 = g;  // keep track of previous g
  for (std::size_t j = 1; const auto & gj : ctrl_points | std::views::drop(1)) {
    const Scalar Btilde = uvec.dot(M.row(j));
    const typename G::Tangent v = (gjm1.inverse() * gj).log();
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
    gjm1 = gj;
    ++j;
  }

  return g;
}


template<LieGroupLike G, std::size_t K = 3>
class BSpline
{
public:
  /**
   * @brief Construct a cardinal bspline defined on [0, 1) with constant value
   */
  BSpline() :
  t0_(0), dt_(1), ctrl_pts_(K+1, G::Identity())
  {}

  /**
   * @brief Create a cardinal BSpline
   * @param t0 start of spline
   * @param dt end of spline
   * @param ctrl_pts spline control points
   *
   * The ctrl_pts - knot points correspondence is as follows
   *
   * KNOT  -K  -K+1   -K+2  ...    0    1   ...  N-K
   * CTRL   0     1      2  ...    K  K+1          N
   *                               ^               ^
   *                             t_min           t_max
   *
   * The first K ctrl_pts are exterior points and are outside
   * the support of the spline, which means that the spline is defined on
   * [t0_, (N-K)*dt]
   *
   * For interpolation purposes use an odd spline degree and set
   *
   *  t0 = (timestamp of first control point) + dt*K/2
   *
   * which aligns control points with the maximum of the corresponding
   * basis function.
   */
  template<std::ranges::range R>
  BSpline(double t0, double dt, R && ctrl_pts)
  requires std::is_same_v<std::ranges::range_value_t<R>, G>
  :
  t0_(t0), dt_(dt), ctrl_pts_(std::forward<R>(ctrl_pts))
  {}

  double t_min() const
  {
    return t0_;
  }

  double t_max() const
  {
    return t0_ + (ctrl_pts_.size() - K) * dt_;
  }

  G eval(
    double t,
    std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
    std::optional<Eigen::Ref<typename G::Tangent>> acc = {}) const
  {
    // index of relevant interval
    int64_t istar = static_cast<int64_t>((t - t0_) / dt_);

    double u;
    // clamp to end of range if necessary
    if (istar < 0) {
      istar = 0;
      u = 0;
    } else if (istar + K + 1 > ctrl_pts_.size()) {
      istar = ctrl_pts_.size() - K - 1;
      u = 1;
    } else {
      u = (t - t0_ - istar * dt_) / dt_;
    }

    G g = bspline_eval<G, K>(
      ctrl_pts_ | std::views::drop(istar) | std::views::take(K+1),
      u, vel, acc
    );

    if (vel.has_value()) {
      vel.value() *= dt_;
    }

    if (acc.has_value()) {
      acc.value() *= dt_ * dt_;
    }

    return g;
  }

private:
  std::vector<G> ctrl_pts_;
  double t0_, dt_;
};


} // namespace smooth
