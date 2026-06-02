#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <../tests/test_util.hpp>
#include <walnuts.hpp>

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

TEST(DetailRandom, StandardNormalReturnMeanAndVariance) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  Eigen::VectorXd v = r.standard_normal(TEST_SIZE);
  double mean = v.mean();
  double var = (v.array() - mean).square().sum() / (TEST_SIZE - 1);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(var, 1.0, 0.05);
}

TEST(DetailRandom, StandardNormalSetMeanAndVariance) {
  std::mt19937 rng(RNG_TEST_SEED);
  walnuts::detail::Random<std::mt19937> r(rng);
  Eigen::VectorXd v;
  r.standard_normal(TEST_SIZE, v);
  double mean = v.mean();
  double var = (v.array() - mean).square().sum() / (TEST_SIZE - 1);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(var, 1.0, 0.05);
}

// log_sum_exp(double, double) function *****************************

TEST(LogSumExp, KnownValues) {
  EXPECT_NEAR(walnuts::detail::log_sum_exp(0.0, 0.0), std::log(2.0), 1e-15);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(-3.0, -3.0), -3.0 + std::log(2.0), 1e-15);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(1000.0, 1000.0), 1000.0 + std::log(2.0), 1e-10);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(1.0, 2.0),
              std::log(std::exp(1.0) + std::exp(2.0)), 1e-15);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(-1.0, -2.0),
              std::log(std::exp(-1.0) + std::exp(-2.0)), 1e-15);
}

TEST(LogSumExp, Symmetry) {
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(1.0, 2.0), walnuts::detail::log_sum_exp(2.0, 1.0));
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(-5.0, 3.0), walnuts::detail::log_sum_exp(3.0, -5.0));
}

TEST(LogSumExp, LargeDifferenceApproximatesMax) {
  EXPECT_NEAR(walnuts::detail::log_sum_exp(1000.0, 0.0), 1000.0, 1e-10);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(0.0, 1000.0), 1000.0, 1e-10);
  EXPECT_NEAR(walnuts::detail::log_sum_exp(-1.0, -1000.0), -1.0, 1e-10);
}

TEST(LogSumExp, NumericalStabilityLargePositive) {
  double result = walnuts::detail::log_sum_exp(1e308, 1e308);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, 1e308 + std::log(2.0), 1e295);
}

TEST(LogSumExp, NumericalStabilityLargeNegative) {
  double result = walnuts::detail::log_sum_exp(-1e308, -1e308);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, -1e308 + std::log(2.0), 1e295);
}

TEST(LogSumExp, InfiniteArgs) {
  double inf = std::numeric_limits<double>::infinity();
  double neg_inf = -inf;
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(neg_inf, 5.0), 5.0);
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(5.0, neg_inf), 5.0);
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(neg_inf, -3.0), -3.0);
  EXPECT_EQ(walnuts::detail::log_sum_exp(neg_inf, neg_inf), neg_inf);

  EXPECT_EQ(walnuts::detail::log_sum_exp(inf, 0.0), inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(0.0, inf), inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(inf, inf), inf);

  EXPECT_TRUE(std::isinf(walnuts::detail::log_sum_exp(inf, neg_inf)));
  EXPECT_TRUE(std::isinf(walnuts::detail::log_sum_exp(neg_inf, inf)));
}

TEST(LogSumExp, NaNPropagates) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(nan, 1.0)));
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(1.0, nan)));
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(nan, nan)));
}

// walnuts::detail::log_sum_exp(VectorXd) function ***********************************

TEST(LogSumExpVector, EmptyVector) {
  Eigen::VectorXd x(0);
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), -std::numeric_limits<double>::infinity());
}

TEST(LogSumExpVector, MatchesTwoArgVersion) {
  Eigen::VectorXd x(2);
  x << 1.0, 2.0;
  EXPECT_NEAR(walnuts::detail::log_sum_exp(x), walnuts::detail::log_sum_exp(1.0, 2.0), 1e-15);
}

// Single element: log(exp(x)) = x
TEST(LogSumExpVector, SingleElement) {
  Eigen::VectorXd x(1);
  x << 3.7;
  EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(x), 3.7);
}

// Equal elements: result is log(n * exp(v)) = v + log(n)
TEST(LogSumExpVector, EqualElements) {
  for (int n : {2, 5, 100}) {
    Eigen::VectorXd x = Eigen::VectorXd::Constant(n, -4.0);
    EXPECT_NEAR(walnuts::detail::log_sum_exp(x), -4.0 + std::log(n), 1e-12);
  }
}

// Known value against naive computation (small values, no overflow risk)
TEST(LogSumExpVector, KnownValueNaive) {
  Eigen::VectorXd x(4);
  x << -1.0, 0.0, 1.0, 2.0;
  double expected = std::log(std::exp(-1.0) + std::exp(0.0)
                             + std::exp(1.0) + std::exp(2.0));
  EXPECT_NEAR(walnuts::detail::log_sum_exp(x), expected, 1e-14);
}

// Numerical stability: large positive values that would overflow naively
TEST(LogSumExpVector, StabilityLargePositive) {
  Eigen::VectorXd x(3);
  x << 1e308, 1e308, 1e308;
  double result = walnuts::detail::log_sum_exp(x);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, 1e308 + std::log(3.0), 1e295);
}

// Numerical stability: large negative values
TEST(LogSumExpVector, StabilityLargeNegative) {
  Eigen::VectorXd x(3);
  x << -1e308, -1e308, -1e308;
  double result = walnuts::detail::log_sum_exp(x);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, -1e308 + std::log(3.0), 1e295);
}

// All -inf -> -inf
TEST(LogSumExpVector, AllNegativeInfinity) {
  double neg_inf = -std::numeric_limits<double>::infinity();
  Eigen::VectorXd x = Eigen::VectorXd::Constant(4, neg_inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), neg_inf);
}

// All +inf -> +inf
TEST(LogSumExpVector, AllPositiveInfinity) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x = Eigen::VectorXd::Constant(4, inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), inf);
}

// One +inf among finite values -> +inf
TEST(LogSumExpVector, OnePositiveInfinity) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x(3);
  x << 1.0, inf, 2.0;
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), inf);
}

// One -inf among finite values: -inf entry contributes nothing
TEST(LogSumExpVector, OneNegativeInfinity) {
  double neg_inf = -std::numeric_limits<double>::infinity();
  Eigen::VectorXd x(3);
  x << 1.0, neg_inf, 2.0;
  Eigen::VectorXd x_without(2);
  x_without << 1.0, 2.0;
  EXPECT_NEAR(walnuts::detail::log_sum_exp(x), walnuts::detail::log_sum_exp(x_without), 1e-15);
}

// Mixed +inf and -inf -> NaN (max is +inf, exp(-inf - inf) = exp(-inf) = 0,
// but exp(+inf - +inf) = exp(NaN), so result is NaN)
TEST(LogSumExpVector, MixedInfinities) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x(3);
  x << inf, 1.0, -inf;
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(x)));
}

// NaN in input propagates
TEST(LogSumExpVector, NaNPropagates) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd x(3);
  x << 1.0, nan, 2.0;
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(x)));
}
