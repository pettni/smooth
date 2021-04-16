static constexpr uint32_t lie_size = Eigen::internal::traits<Derived>::RowsAtCompileTime;
static constexpr uint32_t lie_dof = Eigen::internal::traits<Derived>::RowsAtCompileTime;
static constexpr uint32_t lie_dim = Eigen::internal::traits<Derived>::RowsAtCompileTime + 1;
static constexpr uint32_t lie_actdim = Eigen::internal::traits<Derived>::RowsAtCompileTime;

using Group = Eigen::Matrix<Scalar, lie_size, 1>;
using MatrixGroup = Eigen::Matrix<Scalar, lie_dim, lie_dim>;
using Tangent = Eigen::Matrix<Scalar, lie_dof, 1>;
using TangentMap = Eigen::Matrix<Scalar, lie_dof, lie_dof>;
using Vector = Eigen::Matrix<Scalar, lie_actdim, 1>;

static constexpr bool is_En =
  Derived::IsVectorAtCompileTime == 1 &&
  Derived::RowsAtCompileTime > 0 &&
  Derived::ColsAtCompileTime == 1;

template<typename OtherDerived>
static constexpr bool is_compat =
  is_En &&
  OtherDerived::IsVectorAtCompileTime &&
  OtherDerived::SizeAtCompileTime == lie_size;

void setIdentity() requires(is_En)
{
  static_cast<Derived &>(*this).setZero();
}

template<typename RNG>
void setRandom(RNG & rng) requires(is_En)
{
  std::uniform_real_distribution<Scalar> dist;
  static_cast<Derived &>(*this) = Group::NullaryExpr([&](int) {return dist(rng);});
}

static Group Identity() requires(is_En)
{
  return Group::Zero();
}

template<typename RNG>
static Group Random(RNG &) requires(is_En)
{
  return Group::Zero();
}

MatrixGroup matrix_group() const requires(is_En)
{
  MatrixGroup ret;
  ret.setIdentity();
  ret.template topRightCorner<lie_size, 1>() = static_cast<const Derived &>(*this);
  return ret;
}

// Action works both for group and vectors since they are the same
template<typename OtherDerived>
inline Vector operator*(const Eigen::MatrixBase<OtherDerived> & v) const
requires(is_En && is_compat<OtherDerived>)
{
  return static_cast<const Derived &>(*this) + v;
}

inline Group inverse() const
requires(is_En)
{
  return -static_cast<const Derived &>(*this);
}

inline auto coeffs() const
{
  return static_cast<const Derived &>(*this);
}

inline Tangent log() const
requires(is_En)
{
  return static_cast<const Derived &>(*this);
}

TangentMap Ad() const
requires(is_En)
{
  return TangentMap::Identity();
}

template<typename OtherDerived>
requires(is_En && is_compat<OtherDerived>)
static Group exp(const Eigen::MatrixBase<OtherDerived> & t)
{
  return t;
}

template<typename OtherDerived>
requires(is_En && is_compat<OtherDerived>)
static TangentMap ad(const Eigen::MatrixBase<OtherDerived> & t)
{
  return TangentMap::Zero();
}

template<typename OtherDerived>
requires(is_En && is_compat<OtherDerived>)
static MatrixGroup hat(const Eigen::MatrixBase<OtherDerived> & t)
{
  MatrixGroup ret;
  ret.setZero();
  ret.template topRightCorner<lie_size, 1>() = t;
  return ret;
}

template<typename AlgebraDerived>
requires(is_En)
static Tangent vee(const Eigen::MatrixBase<AlgebraDerived> & a)
{
  return a.template topRightCorner<lie_size, 1>();
}

template<typename TangentDerived>
requires(is_En && is_compat<TangentDerived>)
static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> &)
{
  return TangentMap::Identity();
}

template<typename TangentDerived>
requires(is_En && is_compat<TangentDerived>)
static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> &)
{
  return TangentMap::Identity();
}

template<typename TangentDerived>
requires(is_En && is_compat<TangentDerived>)
static TangentMap dl_exp(const Eigen::MatrixBase<TangentDerived> &)
{
  return TangentMap::Identity();
}

template<typename TangentDerived>
requires(is_En && is_compat<TangentDerived>)
static TangentMap dl_expinv(const Eigen::MatrixBase<TangentDerived> &)
{
  return TangentMap::Identity();
}
