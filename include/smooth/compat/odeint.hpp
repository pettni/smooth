#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

#include "smooth/concepts.hpp"


namespace smooth
{

/**
 * @brief boost::odeint Butcher tableaus are evaluated by weighted
 * calculations of the form y = Σ_{i=1}^n alpha_i x_i which are generically
 * implemented.
 *
 * However, for the special case of butcher tableaus it holds that alpha_1 = 1,
 * and furthermore x_1 is always of the state type while x_2 ... x_n are
 * of the derivative type. The scale sum can therefore be generalized to the Lie
 * group case as
 *
 *   y = x_1 * exp(Σ_{i=2}^n alpha_i x_i)
 *
 * The methods below inject those calculations into boost:odeint to enable
 * numerical integration on Lie groups. For succintness we implement a single
 * method using variadic templates.
 */
struct lie_operations
{
  template<typename ... Fac>
  struct scale_sum
  {
    const std::tuple<Fac ...> m_alpha;

    scale_sum(Fac ... alpha)
    : m_alpha(alpha ...)
    {
      if (std::get<0>(m_alpha) != std::tuple_element_t<0, std::tuple<Fac...>>(1)) {
        throw std::runtime_error("lie_operations only valid for alpha1 = 1");
      }
    }

    template<typename ... Ts, std::size_t ... Is>
    auto helper(std::index_sequence<Is...>, const Ts & ... as)
    {
      // plus 1 since alpha1 = 1 is not included in Ts...
      return ((std::get<Is + 1>(m_alpha) * as) + ...);
    }

    template<LieGroupLike T1, LieGroupLike T2, typename ... Ts>
    requires std::is_same_v<T1, T2>&&
    std::conjunction_v<std::is_same<typename T1::Tangent, Ts>...>
    void operator()(T1 & y, const T2 & x, const Ts & ... as)
    {
      y = x * T2::exp(helper(std::make_index_sequence<sizeof...(Ts)>(), as...));
    }

    using result_type = void;
  };

  template<typename Fac1, typename Fac2 = Fac1>
  using scale_sum2 = scale_sum<Fac1, Fac2>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2>
  using scale_sum3 = scale_sum<Fac1, Fac2, Fac3>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3>
  using scale_sum4 = scale_sum<Fac1, Fac2, Fac3, Fac4>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4>
  using scale_sum5 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5>
  using scale_sum6 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6>
  using scale_sum7 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7>
  using scale_sum8 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8>
  using scale_sum9 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9>
  using scale_sum10 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10>
  using scale_sum11 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11>
  using scale_sum12 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11, typename Fac13 = Fac12>
  using scale_sum13 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12, Fac13>;

  template<typename Fac1, typename Fac2 = Fac1, typename Fac3 = Fac2, typename Fac4 = Fac3, typename Fac5 = Fac4, typename Fac6 = Fac5, typename Fac7 = Fac6, typename Fac8 = Fac7, typename Fac9 = Fac8, typename Fac10 = Fac9, typename Fac11 = Fac10, typename Fac12 = Fac11, typename Fac13 = Fac12, typename Fac14 = Fac13>
  using scale_sum14 = scale_sum<Fac1, Fac2, Fac3, Fac4, Fac5, Fac6, Fac7, Fac8, Fac9, Fac10, Fac11, Fac12, Fac13, Fac14>;
};

}  // namespace smooth


// Specialize internal boost trait to use the above operations for Lie
// group types. Note that boost::numeric::odeint::vector_space_algebra
// must be specified for the steppers.
template<::smooth::LieGroupLike G>
struct boost::numeric::odeint::operations_dispatcher_sfinae<G, void>
{
  using operations_type = ::smooth::lie_operations;
};
