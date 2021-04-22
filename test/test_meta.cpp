#include <gtest/gtest.h>

#include "smooth/meta.hpp"


using namespace smooth::meta;


TEST(Meta, StaticFor)
{

}

TEST(Meta, IndexSequence)
{
  using iseq = std::index_sequence<3, 5, 7, 2, 4>;

  static_assert(
    std::is_same_v<
      iseq_add_t<5, iseq>,
      std::index_sequence<8, 10, 12, 7, 9>
    >
  );

  static_assert(iseq_sum_v<iseq> == 21);

  static_assert(iseq_el_v<0, iseq> == 3);
  static_assert(iseq_el_v<1, iseq> == 5);
  static_assert(iseq_el_v<2, iseq> == 7);
  static_assert(iseq_el_v<3, iseq> == 2);
  static_assert(iseq_el_v<4, iseq> == 4);

  static_assert(iseq_len_v<iseq> == 5);

  static_assert(
    std::is_same_v<
      iseq_psum_t<iseq>,
      std::index_sequence<0, 3, 8, 15, 17>
    >
  );
}

TEST(Meta, TypePack)
{
  using pack1 = typepack<float, double, uint32_t, std::string>;
  using pack2 = typepack<std::size_t, uint64_t>;

  static_assert(
    std::is_same_v<
      typepack_cat_t<pack1, pack2>,
      typepack<float, double, uint32_t, std::string, std::size_t, uint64_t>
    >
  );

  static_assert(
    std::is_same_v<
      typepack_cat_t<pack2, pack2, pack2>,
      typepack<std::size_t, uint64_t, std::size_t, uint64_t, std::size_t, uint64_t>
    >
  );

  static_assert(
    std::is_same_v<
      typepack_take_t<3, pack1>,
      typepack<float, double, uint32_t>
    >
  );

  static_assert(
    std::is_same_v<
      typepack_drop_t<2, pack1>,
      typepack<uint32_t, std::string>
    >
  );

  static_assert(
    std::is_same_v<
      pack1::apply<std::tuple>,
      std::tuple<float, double, uint32_t, std::string>
    >
  );
}

TEST(Meta, ChangeTemplateArg)
{
  using tuple_t = std::tuple<float, double, uint32_t, std::string>;

  static_assert(
    std::is_same_v<
      change_template_arg_t<tuple_t, 2, double>,
      std::tuple<float, double, double, std::string>
    >
  );

  static_assert(
    std::is_same_v<
      change_template_arg_t<change_template_arg_t<tuple_t, 2, double>, 3, int>,
      std::tuple<float, double, double, int>
    >
  );
}
