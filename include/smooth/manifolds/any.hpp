// Copyright (C) 2023 Petter Nilsson. MIT License.

#pragma once

#include <memory>
#include <utility>

#include "smooth/concepts/manifold.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief Type erasure Manifold class.
 */
class AnyManifold
{
public:
  /// @brief Constructor.
  AnyManifold() { throw std::runtime_error("Can not default-construct"); }

  /// @brief Construct from typed manifoild
  template<typename M>
  explicit AnyManifold(const M & m) : m_val(std::make_unique<wrapper<std::decay_t<M>>>(m))
  {}

  /// @brief Copy constructor.
  AnyManifold(const AnyManifold & m) : m_val(m.m_val->clone()) {}

  /// @brief Move constructor.
  AnyManifold(AnyManifold && m) : m_val(std::move(m.m_val)) {}

  /// @brief Copy assignment.
  AnyManifold & operator=(const AnyManifold & m)
  {
    m_val = m.m_val->clone();
    return *this;
  }

  /// @brief Move assignment.
  AnyManifold & operator=(AnyManifold && m)
  {
    m_val = std::move(m.m_val);
    return *this;
  }

  /// @brief Get value (mutable).
  template<Manifold M>
  M & get()
  {
    return static_cast<wrapper<M> *>(m_val.get())->get();
  }

  /// @brief Get value (const).
  template<Manifold M>
  const M & get() const
  {
    return static_cast<const wrapper<M> *>(m_val.get())->get();
  }

  /// @brief Degrees of freedom.
  Eigen::Index dof() const { return m_val->dof(); }

  /// @brief Right-plus.
  AnyManifold rplus(Eigen::Ref<const Eigen::VectorXd> a) const { return AnyManifold(m_val->rplus(a)); }

  /// @brief Right-minus.
  Eigen::VectorXd rminus(const AnyManifold & m2) const { return m_val->rminus(m2.m_val); }

private:
  class wrapper_base
  {
  public:
    virtual ~wrapper_base()                                                              = default;
    virtual Eigen::Index dof() const                                                     = 0;
    virtual std::unique_ptr<wrapper_base> rplus(Eigen::Ref<const Eigen::VectorXd>) const = 0;
    virtual Eigen::VectorXd rminus(const std::unique_ptr<wrapper_base> & o) const        = 0;
    virtual std::unique_ptr<wrapper_base> clone() const                                  = 0;
  };

  template<Manifold M>
  class wrapper : public wrapper_base
  {
  public:
    explicit wrapper(const M & val) : m_val(val) {}
    explicit wrapper(M && val) : m_val(std::move(val)) {}
    Eigen::Index dof() const override { return ::smooth::dof(m_val); }
    std::unique_ptr<wrapper_base> rplus(Eigen::Ref<const Eigen::VectorXd> a) const override
    {
      return std::make_unique<wrapper<M>>(::smooth::rplus(m_val, a));
    }
    Eigen::VectorXd rminus(const std::unique_ptr<wrapper_base> & o) const override
    {
      return ::smooth::rminus(m_val, static_cast<const wrapper<M> *>(o.get())->m_val);
    }
    std::unique_ptr<wrapper_base> clone() const override { return std::make_unique<wrapper<M>>(m_val); }
    M & get() { return m_val; }
    const M & get() const { return m_val; }

  private:
    M m_val;
  };

private:
  explicit AnyManifold(std::unique_ptr<wrapper_base> val) : m_val(std::move(val)) {}
  std::unique_ptr<wrapper_base> m_val;
};

namespace traits {

template<>
struct man<AnyManifold>
{
  using Scalar      = double;
  using PlainObject = AnyManifold;
  template<typename NewScalar>
  using CastT = PlainObject;

  static constexpr int Dof = -1;

  static inline Eigen::Index dof(const PlainObject & m) { return m.dof(); }

  static inline PlainObject Default(Eigen::Index) { throw std::runtime_error("AnyManifold: default not supported"); }

  template<typename NewScalar>
  static inline CastT<NewScalar> cast(const PlainObject &)
  {
    throw std::runtime_error("AnyManifold: cast not supported");
  }

  template<typename Derived>
  static inline PlainObject rplus(const PlainObject & m, const Eigen::MatrixBase<Derived> & a)
  {
    return m.rplus(a);
  }

  static inline Eigen::Vector<Scalar, Dof> rminus(const PlainObject & m1, const PlainObject & m2)
  {
    return m1.rminus(m2);
  }
};

}  // namespace traits

SMOOTH_END_NAMESPACE
