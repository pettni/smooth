#ifndef SMOOTH__LIE_GROUP_BASE_HPP_
#define SMOOTH__LIE_GROUP_BASE_HPP_

#include <Eigen/Core>


namespace smooth
{

// required to provide storage
template<typename S>
concept StorageC =
requires {
  typename S::value_type;
  typename S::size_type;
}
&& requires(const S & s, typename S::size_type i) {
  { s[i] } -> std::same_as<const typename S::value_type &>;
};

// required to be a group implementation
template<typename G>
concept LieGroupC =
std::is_same_v<std::decay_t<decltype(G::n)>, uint32_t>
&& std::is_same_v<std::decay_t<decltype(G::dof)>, uint32_t>;

// required for a valid group-storage pair
template<typename G, typename S>
concept Groupable = LieGroupC<G> && StorageC<S>
&& sizeof(S) == sizeof(typename S::value_type) * G::n
&& requires(const S & s) {
  { G::log(s) } -> std::same_as<Eigen::Matrix<typename S::value_type, G::dof, 1>>;
};


struct SO3
{
  static constexpr uint32_t n = 4;
  static constexpr uint32_t dof = 3;

  template<typename S>
  static Eigen::Matrix<double, dof, 1> log(const S & s) {
    return Eigen::Matrix<double, dof, 1>(s[0], s[1], s[2]);
  }
};


// se3 storage
// ROS message:       x y z qx qy qz qw
// Eigen quaternion:  qx qy qz qw
template<typename G, typename S> requires Groupable<G, S>
class LieGroup
{
public:
  using Tangent = Eigen::Matrix<double, G::dof, 1>;

  LieGroup() = default;

  // constructor from storage
  LieGroup(const S & s)
    : s_(s)
  {}

  LieGroup(S && s)
    : s_(std::move(s))
  {}

  // copy constructor from other storage
  template<typename SOther>
  LieGroup(const LieGroup<G, SOther> & o) {
    for (auto i = 0u; i != G::dof; ++i) {
      s_[i] = o.storage()[i];
    }
  }

  // copy assignment from other storage
  template<typename SOther>
  LieGroup & operator=(const LieGroup<G, SOther> & o) {
    // TODO: static for
    for (auto i = 0u; i != G::dof; ++i) {
      s_[i] = o.storage()[i];
    }
    return *this;
  }

  Tangent log() const
  {
    return G::log(s_);
  }

  S & storage() {
    return s_;
  }

  const S & storage() const {
    return s_;
  }

private:
  S s_;
};


// example alternative storage
template<typename T, std::size_t N>
struct ReverseStorage
{
  static constexpr std::size_t size = N;
  using value_type = T;
  using size_type = std::size_t;

  constexpr T & operator[](size_type i) {
    return data[3-i];
  }

  constexpr const T & operator[](size_type i) const {
    return data[3-i];
  }

  std::array<T, N> data;
};


// typedefs with regular storage
using SO3d = LieGroup<SO3, std::array<double, 4>>;
using SO3rev = LieGroup<SO3, ReverseStorage<double, 4>>;


}  // namespace smooth

#endif  // SMOOTH__LIE_GROUP_BASE_HPP_

