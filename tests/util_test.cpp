#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <../tests/test_util.hpp>
#include <walnuts.hpp>

// Constants ********************************************************

// FALSE_SHARING_GUARD_SIZE constant

TEST(Detail, FalseSharingGuardSizeIs128) {
  EXPECT_EQ(walnuts::detail::FALSE_SHARING_GUARD_SIZE, std::size_t{128});
}

// Update enums

TEST(Detail, UpdateEnumValuesAreDistinct) {
  EXPECT_NE(walnuts::detail::Update::Barker,
            walnuts::detail::Update::Metropolis);
}

// Direction enums

TEST(Detail, DirectionEnumValuesAreDistinct) {
  EXPECT_NE(walnuts::detail::Direction::Forward,
            walnuts::detail::Direction::Backward);
}

// Direction types

TEST(Detail, ForwardAndBackwardTypesCarryCorrectValues) {
  EXPECT_EQ(walnuts::detail::Backward_t::value,
            walnuts::detail::Direction::Backward);
  EXPECT_EQ(walnuts::detail::Forward_t::value,
            walnuts::detail::Direction::Forward);
}

TEST(Detail, ForwardAndBackwardTypesAreDistinct) {
  static_assert(
      !std::is_same_v<walnuts::detail::Forward_t, walnuts::detail::Backward_t>);
}

// Random class *****************************************************

constexpr std::size_t RNG_TEST_SEED = 12345;
constexpr std::size_t TEST_SIZE = 100000;

// construction

TEST(DetailRandom, RngStateAdvancesThroughWrapper) {
  // Generating through the wrapper must consume the underlying engine
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  auto state_before = rng;
  static_cast<void>(r.uniform_real_01());
  EXPECT_NE(rng, state_before);
}

// uniform_real_01() method

TEST(DetailRandom, UniformReal01InRange) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  for (int i = 0; i < TEST_SIZE; ++i) {
    double u = r.uniform_real_01();
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

TEST(DetailRandom, UniformReal01MeanAndVariance) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < TEST_SIZE; ++i) {
    double u = r.uniform_real_01();
    sum += u;
    sum_sq += u * u;
  }
  double mean = sum / TEST_SIZE;
  double var = sum_sq / TEST_SIZE - mean * mean;
  EXPECT_NEAR(mean, 0.5, 0.01);        // 10 se tolerance
  EXPECT_NEAR(var, 1.0 / 12.0, 0.01);  // 8 se tolerance
}

// rng() method

TEST(DetailRandom, RngHeldByReference) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  EXPECT_EQ(&r.rng(), &rng);
}

// uniform_binary() method

TEST(DetailRandom, UniformBinaryReturnsBoolWithCorrectMean) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  int true_count = 0;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (r.uniform_binary()) {
      ++true_count;
    }
  }
  double p_hat = static_cast<double>(true_count) / TEST_SIZE;
  EXPECT_NEAR(p_hat, 0.5, 0.02);  // 10 se threshold
}

// standard_normal() method

TEST(DetailRandom, StandardNormalReturnsCorrectSize) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  Eigen::VectorXd v = r.standard_normal(7);
  EXPECT_EQ(v.size(), Eigen::Index{7});
}

TEST(DetailRandom, StandardNormalOutVariantWritesCorrectSize) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  Eigen::VectorXd out;
  r.standard_normal(5, out);
  EXPECT_EQ(out.size(), Eigen::Index{5});
}

TEST(DetailRandom, StandardNormalMeanAndVariance) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  Eigen::VectorXd v = r.standard_normal(TEST_SIZE);
  double mean = v.mean();
  double var = (v.array() - mean).square().sum() / (TEST_SIZE - 1);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(var, 1.0, 0.05);
}
