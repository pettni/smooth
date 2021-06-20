#include "lie_group.hpp"

class SO3Tag
{};

static constexpr double eps2 = 1e-8;

template<>
struct lie_impl<SO3Tag>
{
  static constexpr Eigen::Index Dof     = 3;
  static constexpr Eigen::Index RepSize = 4;

  template<typename Derived>
  static void exp(
    const Eigen::MatrixBase<Derived> & a,
    Eigen::Ref<Eigen::Array<typename Derived::Scalar, RepSize, 1>> s
  )
  {
    using Scalar = typename Derived::Scalar;
    using std::sqrt, std::cos, std::sin;

    const Scalar th2 = a.squaredNorm();

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+sin%28x%2F2%29%2Fx+at+x%3D0
      A = Scalar(1) / Scalar(2) - th2 / Scalar(48);
      // https://www.wolframalpha.com/input/?i=series+cos%28x%2F2%29+at+x%3D0
      B = Scalar(1) - th2 / Scalar(8);
    } else {
      const Scalar th = sqrt(th2);
      A               = sin(th / Scalar(2)) / th;
      B               = cos(th / Scalar(2));
    }

    s = Eigen::Matrix<Scalar, RepSize, 1>(B, A * a.x(), A * a.y(), A * a.z());
  }

  template<typename Derived>
  static void log(
    const Eigen::ArrayBase<Derived> & s,
    Eigen::Ref<Eigen::Matrix<typename Derived::Scalar, Dof, 1>> a
  )
  {
    using Scalar = typename Derived::Scalar;

    using std::atan2, std::sqrt;
    const Scalar xyz2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];

    Scalar phi;
    if (xyz2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+atan%28y%2Fx%29+%2F+y+at+y%3D0
      phi = Scalar(2) / s[3] - Scalar(2) * xyz2 / (Scalar(3) * s[3] * s[3] * s[3]);
    } else {
      Scalar xyz = sqrt(xyz2);
      phi        = Scalar(2) * atan2(xyz, s[3]) / xyz;
    }
    a = phi * Eigen::Matrix<Scalar, 3, 1>(s[0], s[1], s[2]);
  }
};

template<typename Scalar, typename Coeffs = Eigen::Array<Scalar, 4, 1>>
class SO3: public LieGroup<Scalar, SO3Tag, Coeffs>
{
public:
  using Base = LieGroup<Scalar, SO3Tag>;
  using Base::operator=;

  SO3() = default;

  template<typename C>
  SO3(C && c)
  : Base(std::forward<C>(c)) {}

  SO3(const Eigen::Quaternion<Scalar> & quat) { Base::c_.coeffs() = quat.coeffs(); }

  Eigen::Map<Eigen::Quaternion<Scalar>> & quat()
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(Base::c_.data());
  }

  Eigen::Map<const Eigen::Quaternion<Scalar>> & quat() const
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(Base::c_.data());
  }
};

template<typename Scalar>
using SO3Map = SO3<Scalar, Eigen::Map<Eigen::Array<Scalar, 4, 1>>>;

template<typename Scalar>
using SO3ConstMap = SO3<Scalar, Eigen::Map<const Eigen::Array<Scalar, 4, 1>>>;

