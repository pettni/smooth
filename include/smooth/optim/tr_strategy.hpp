// Copyright (C) 2023 Petter Nilsson. MIT License.

#pragma once

#include <memory>

#include "smooth/version.hpp"

SMOOTH_BEGIN_NAMESPACE

class TrustRegionStrategy
{
public:
  virtual ~TrustRegionStrategy() = default;
  /// @brief Get trust region size.
  virtual double get_delta() const = 0;
  /// @brief Update trust region and determine if step is taken.
  virtual bool step_and_update(const double rho) = 0;
};

/**
 * @brief Trust region strategy used in the Ceres solver.
 *
 * https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/levenberg_marquardt_strategy.cc
 */
class CeresStrategy : public TrustRegionStrategy
{
public:
  inline double get_delta() const override { return m_delta; }
  inline bool step_and_update(const double rho) override
  {
    if (rho > 1e-3) {
      const double two_rho_min_1 = 2 * rho - 1;
      m_delta /= std::max(1. / 3, 1 - two_rho_min_1 * two_rho_min_1 * two_rho_min_1);
      m_reduce = 2;
      return true;
    } else {
      m_delta /= m_reduce;
      m_reduce *= 2;
      return false;
    }
  }

private:
  double m_delta{10000};
  double m_reduce{2};
};

/**
 * @brief Trust region strategy used in:
 *
 * Fast Nonlinear Least Squares Optimization of Large-Scale Semi-Sparse Problems
 */
class DisneyStrategy : public TrustRegionStrategy
{
public:
  inline double get_delta() const override { return m_delta; }

  inline bool step_and_update(const double rho) override
  {
    if (rho > 0) {
      m_delta = 1000;
      return true;
    } else {
      m_delta /= 10;
      return false;
    }
  }

private:
  double m_delta{1000};
};

SMOOTH_END_NAMESPACE
