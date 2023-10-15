// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <algorithm>
#include <cassert>

#include <Eigen/Core>

#include "../../polynomial/basis.hpp"
#include "../cumulative_spline.hpp"
#include "../spline.hpp"

SMOOTH_BEGIN_NAMESPACE

// cumulative basis functions
template<int K>
inline static constexpr auto kBasisFunction = polynomial_cumulative_basis<PolynomialBasis::Bernstein, K, double>();

// mapped cumulative basis functions
template<int K>
inline static const Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> kMappedBasisFunction =
  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(kBasisFunction<K>[0].data());

template<int K, LieGroup G>
Spline<K, G>::Spline(const G & ga) : m_g0{ga}, m_end_t{}, m_end_g{}, m_Vs{}, m_seg_T0{}, m_seg_Del{}
{}

template<int K, LieGroup G>
Spline<K, G>::Spline(double T, Eigen::Matrix<double, Dof<G>, K> && V, G && ga)
    : m_g0{std::move(ga)}, m_end_t{T}, m_Vs{{std::move(V)}}, m_seg_T0{0}, m_seg_Del{1}
{
  assert(T > 0);

  m_end_g.resize(1);
  if constexpr (K == 0) {
    m_end_g[0] = m_g0;
  } else {
    m_end_g[0] = composition(m_g0, cspline_eval_vs<K, G>(m_Vs[0].colwise(), kMappedBasisFunction<K>, 1.));
  }
}

template<int K, LieGroup G>
template<typename Derived>
Spline<K, G>::Spline(double T, const Eigen::MatrixBase<Derived> & V, const G & ga)
    : Spline(T, Eigen::Matrix<double, Dof<G>, K>(V), G(ga))
{}

template<int K, LieGroup G>
template<std::ranges::range Rv>
  requires(std::is_same_v<std::ranges::range_value_t<Rv>, Tangent<G>>)
Spline<K, G>::Spline(double T, const Rv & vs, const G & ga) : m_g0(ga), m_end_t{T}, m_seg_T0{0}, m_seg_Del{1}
{
  assert(T > 0);
  assert(std::ranges::size(vs) == K);

  m_Vs.resize(1);
  for (auto i = 0u; const auto & v : vs) { m_Vs[0].col(i++) = v; }

  m_end_g.resize(1);
  if constexpr (K == 0) {
    m_end_g[0] = m_g0;
  } else {
    m_end_g[0] = composition(m_g0, cspline_eval_vs<K, G>(m_Vs[0].colwise(), kMappedBasisFunction<K>, 1.));
  }
}

template<int K, LieGroup G>
Spline<K, G> Spline<K, G>::ConstantVelocityGoal(const G & gb, double T, const G & ga)
{
  assert(T > 0);
  return ConstantVelocity((gb - ga) / T, T, ga);
}

template<int K, LieGroup G>
Spline<K, G> Spline<K, G>::ConstantVelocity(const Tangent<G> & v, double T, const G & ga)
{
  if (T <= 0) {
    return Spline();
  } else {
    Eigen::Matrix<double, Dof<G>, K> V = (T / 3) * v.replicate(1, K);
    return Spline(T, std::move(V), ga);
  }
}

template<int K, LieGroup G>
Spline<K, G>
Spline<K, G>::FixedCubic(const G & gb, const Tangent<G> & va, const Tangent<G> & vb, double T, const G & ga)
  requires(K == 3)
{
  Eigen::Matrix<double, Dof<G>, K> V;
  V.col(0) = T * va / 3;
  V.col(2) = T * vb / 3;
  V.col(1) = log(composition(::smooth::exp<G>(-V.col(0)), composition(inverse(ga), gb), ::smooth::exp<G>(-V.col(2))));
  return Spline(T, std::move(V), ga);
}

template<int K, LieGroup G>
std::size_t Spline<K, G>::size() const
{
  return m_end_t.size();
}

template<int K, LieGroup G>
bool Spline<K, G>::empty() const
{
  return size() == 0;
}

template<int K, LieGroup G>
void Spline<K, G>::reserve(std::size_t capacity)
{
  m_end_t.reserve(capacity);
  m_end_g.reserve(capacity);
  m_Vs.reserve(capacity);
  m_seg_T0.reserve(capacity);
  m_seg_Del.reserve(capacity);
}

template<int K, LieGroup G>
double Spline<K, G>::t_min() const
{
  return 0;
}

template<int K, LieGroup G>
double Spline<K, G>::t_max() const
{
  if (empty()) { return 0; }
  return m_end_t.back();
}

template<int K, LieGroup G>
G Spline<K, G>::start() const
{
  return m_g0;
}

template<int K, LieGroup G>
G Spline<K, G>::end() const
{
  if (empty()) { return m_g0; }
  return m_end_g.back();
}

template<int K, LieGroup G>
void Spline<K, G>::make_local()
{
  m_g0 = Identity<G>();
}

template<int K, LieGroup G>
Spline<K, G> & Spline<K, G>::concat_global(const Spline & other)
{
  const std::size_t N1 = size();
  const std::size_t N2 = other.size();

  const double tend = t_max();

  if (empty()) {
    m_g0 = other.m_g0;
  } else {
    m_end_g[N1 - 1] = other.m_g0;
  }

  m_end_t.resize(N1 + N2);
  m_end_g.resize(N1 + N2);
  m_Vs.resize(N1 + N2);
  m_seg_T0.resize(N1 + N2);
  m_seg_Del.resize(N1 + N2);

  for (auto i = 0u; i < N2; ++i) {
    m_end_t[N1 + i]   = tend + other.m_end_t[i];
    m_end_g[N1 + i]   = other.m_end_g[i];
    m_Vs[N1 + i]      = other.m_Vs[i];
    m_seg_T0[N1 + i]  = other.m_seg_T0[i];
    m_seg_Del[N1 + i] = other.m_seg_Del[i];
  }

  return *this;
}

template<int K, LieGroup G>
Spline<K, G> & Spline<K, G>::concat_local(const Spline & other)
{
  std::size_t N1 = size();
  std::size_t N2 = other.size();

  const double tend = t_max();
  const G gend      = end();

  if (empty()) {
    m_g0 = composition(m_g0, other.m_g0);
  } else {
    m_end_g.back() = composition(m_end_g.back(), other.m_g0);
  }

  m_end_t.resize(N1 + N2);
  m_end_g.resize(N1 + N2);
  m_Vs.resize(N1 + N2);
  m_seg_T0.resize(N1 + N2);
  m_seg_Del.resize(N1 + N2);

  for (auto i = 0u; i < N2; ++i) {
    m_end_t[N1 + i]   = tend + other.m_end_t[i];
    m_end_g[N1 + i]   = composition(gend, other.m_end_g[i]);
    m_Vs[N1 + i]      = other.m_Vs[i];
    m_seg_T0[N1 + i]  = other.m_seg_T0[i];
    m_seg_Del[N1 + i] = other.m_seg_Del[i];
  }

  return *this;
}

template<int K, LieGroup G>
Spline<K, G> & Spline<K, G>::operator+=(const Spline & other)
{
  return concat_local(other);
}

template<int K, LieGroup G>
Spline<K, G> Spline<K, G>::operator+(const Spline & other)
{
  Spline ret = *this;
  ret += other;
  return ret;
}

template<int K, LieGroup G>
template<typename S>
CastT<S, G> Spline<K, G>::operator()(const S & t, OptTangent<CastT<S, G>> vel, OptTangent<CastT<S, G>> acc) const
{
  if (empty() || t < S(0)) {
    if (vel.has_value()) { vel.value().setZero(); }
    if (acc.has_value()) { acc.value().setZero(); }
    return cast<S>(m_g0);
  } else if (t > S(t_max())) {
    if (vel.has_value()) { vel.value().setZero(); }
    if (acc.has_value()) { acc.value().setZero(); }
    return cast<S>(m_end_g.back());
  }

  const auto istar = find_idx(static_cast<double>(t));

  const double ta = istar == 0 ? 0 : m_end_t[istar - 1];
  const double T  = m_end_t[istar] - ta;

  const double Del = m_seg_Del[istar];
  const S u        = std::clamp<S>(S(m_seg_T0[istar]) + S(Del) * (t - S(ta)) / S(T), S(0.), S(1.));

  G g0 = istar == 0 ? m_g0 : m_end_g[istar - 1];

  if constexpr (K == 0) {
    // piecewise constant, nothing to evaluate
    if (vel.has_value()) { vel.value().setZero(); }
    if (acc.has_value()) { acc.value().setZero(); }
    return g0;
  } else {
    // compensate for cropped intervals
    if (m_seg_T0[istar] > 0) {
      g0 = composition(
        g0, inverse(cspline_eval_vs<K, G>(m_Vs[istar].colwise(), kMappedBasisFunction<K>, m_seg_T0[istar])));
    }
    const CastT<S, G> add = cspline_eval_vs<K, CastT<S, G>>(
      m_Vs[istar].template cast<S>().colwise(), kMappedBasisFunction<K>.template cast<S>(), u, vel, acc);
    if (vel.has_value()) { vel.value() *= S(Del / T); }
    if (acc.has_value()) { acc.value() *= S(Del * Del / (T * T)); }
    return composition(cast<S>(g0), add);
  }
}

template<int K, LieGroup G>
Tangent<G> Spline<K, G>::arclength(double t) const
  requires(K == 3)
{
  Tangent<G> ret = Tangent<G>::Zero();

  for (auto i = 0u; i < m_end_t.size(); ++i) {
    // check if we have reached t
    if (i > 0 && t <= m_end_t[i - 1]) { break; }

    // polynomial coefficients a0 + a1 x + a2 x2 + a3 x3
    const Eigen::Matrix<double, K + 1, Dof<G>> coefs = kMappedBasisFunction<K>.rightCols(K) * m_Vs[i].transpose();

    const double ta = i == 0 ? 0 : m_end_t[i - 1];
    const double tb = m_end_t[i];

    const double ua = m_seg_T0[i];
    const double ub = ua + m_seg_Del[i] * (std::min<double>(t, tb) - ta) / (tb - ta);

    for (auto k = 0u; k < Dof<G>; ++k) {
      // derivative b0 + b1 x + b2 x2 has coefficients [b0, b1, b2] = [a1, 2a2, 3a3]
      ret(k) += integrate_absolute_polynomial(ua, ub, 3 * coefs(3, k), 2 * coefs(2, k), coefs(1, k));
    }
  }

  return ret;
}

template<int K, LieGroup G>
Spline<K, G> Spline<K, G>::crop(double ta, double tb, bool localize) const
{
  ta = std::max<double>(ta, 0);
  tb = std::min<double>(tb, t_max());

  if (tb <= ta) { return Spline(); }

  const std::size_t i0 = find_idx(ta);
  std::size_t Nseg     = find_idx(tb) + 1 - i0;

  // prevent last segment from being empty
  if (Nseg >= 2 && m_end_t[i0 + Nseg - 2] == tb) { --Nseg; }

  if (Nseg == 0) { return Spline(); }  // appease santizer

  // state at new from beginning of Spline
  G ga = operator()(ta);

  std::vector<double> end_t(Nseg);
  std::vector<G> end_g(Nseg);
  std::vector<Eigen::Matrix<double, Dof<G>, K>> vs(Nseg);
  std::vector<double> seg_T0(Nseg), seg_Del(Nseg);

  // copy over all relevant segments
  for (auto i = 0u; i < Nseg; ++i) {
    if (i == Nseg - 1) {
      end_t[i] = tb - ta;
      end_g[i] = composition(inverse(ga), operator()(tb));
    } else {
      end_t[i] = m_end_t[i0 + i] - ta;
      end_g[i] = composition(inverse(ga), m_end_g[i0 + i]);
    }
    vs[i]      = m_Vs[i0 + i];
    seg_T0[i]  = m_seg_T0[i0 + i];
    seg_Del[i] = m_seg_Del[i0 + i];
  }

  // crop first segment
  {
    const double tta = 0;
    const double ttb = m_end_t[i0];
    const double sa  = ta;
    const double sb  = ttb;

    seg_T0[0] += seg_Del[0] * (sa - tta) / (ttb - tta);
    seg_Del[0] *= (sb - sa) / (ttb - tta);
  }

  // crop last segment
  {
    const double tta = Nseg == 1 ? ta : m_end_t[Nseg - 2];
    const double ttb = m_end_t[Nseg - 1];
    const double sa  = tta;
    const double sb  = tb;

    seg_T0[Nseg - 1] += seg_Del[Nseg - 1] * (sa - tta) / (ttb - tta);
    seg_Del[Nseg - 1] *= (sb - sa) / (ttb - tta);
  }

  // create new Spline with appropriate body velocities
  Spline<K, G> ret;
  ret.m_g0      = localize ? Identity<G>() : std::move(ga);
  ret.m_end_t   = std::move(end_t);
  ret.m_end_g   = std::move(end_g);
  ret.m_Vs      = std::move(vs);
  ret.m_seg_T0  = std::move(seg_T0);
  ret.m_seg_Del = std::move(seg_Del);
  return ret;
}

template<int K, LieGroup G>
std::size_t Spline<K, G>::find_idx(double t) const
{
  // target condition:
  //  m_end_t[istar - 1] <= t < m_end_t[istar]

  std::size_t istar = 0;

  auto it = utils::binary_interval_search(m_end_t, t);
  if (it != m_end_t.end()) {
    istar = std::min(static_cast<std::size_t>(std::distance(m_end_t.begin(), it)) + 1, m_end_t.size() - 1);
  }

  return istar;
}

SMOOTH_END_NAMESPACE
