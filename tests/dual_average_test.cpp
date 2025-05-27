#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <walnuts/dual_average.hpp>

TEST(DualAvg, one) {
  EXPECT_EQ(2, 1 + 1);
}
