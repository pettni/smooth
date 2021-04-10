#include <gtest/gtest.h>

#include "smooth/base.hpp"


TEST(Base, Base)
{
  smooth::SO3d g{};

  g.storage()[0] = 1;
  g.storage()[1] = 2;
  g.storage()[2] = -2;
  g.storage()[3] = 3;

  std::cout << g.log() << std::endl;

  smooth::ReverseStorage<double, 4> srev;
  srev.data[0] = 5;
  srev.data[1] = 6;
  srev.data[2] = 7;
  srev.data[3] = 8;

  smooth::SO3rev grev(srev);

  std::cout << grev.log() << std::endl;

  g = grev;

  std::cout << g.log() << std::endl;
};
