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

// walnuts::detail::log_sum_exp(VectorXd) function ******************

TEST(LogSumExpVector, EmptyVector) {
  Eigen::VectorXd x(0);
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), -std::numeric_limits<double>::infinity());
}

TEST(LogSumExpVector, SingleElement) {
  for (auto x : std::vector<double>{-1000.0, 0.0, 3.7, 1000.2}) {
    Eigen::VectorXd v(1);
    v << x;
    EXPECT_DOUBLE_EQ(walnuts::detail::log_sum_exp(v), x);
  }
}

TEST(LogSumExpVector, EqualElements) {
  for (int n : {2, 5, 100}) {
    Eigen::VectorXd x = Eigen::VectorXd::Constant(n, -4.0);
    EXPECT_NEAR(walnuts::detail::log_sum_exp(x), -4.0 + std::log(n), 1e-12);
  }
}

TEST(LogSumExpVector, StabilityLargePositive) {
  Eigen::VectorXd x(3);
  x << 1e308, 1e308, 1e308;
  double result = walnuts::detail::log_sum_exp(x);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, 1e308 + std::log(3.0), 1e295);
}

TEST(LogSumExpVector, StabilityLargeNegative) {
  Eigen::VectorXd x(3);
  x << -1e308, -1e308, -1e308;
  double result = walnuts::detail::log_sum_exp(x);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_NEAR(result, -1e308 + std::log(3.0), 1e295);
}

TEST(LogSumExpVector, AllInfinity) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x = Eigen::VectorXd::Constant(4, inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), inf);
  EXPECT_EQ(walnuts::detail::log_sum_exp(-x), -inf);
  
}

TEST(LogSumExpVector, OnePositiveInfinity) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x(3);
  x << 1.0, inf, 2.0;
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), inf);
  x << 1.0, -inf, 2.0;
  EXPECT_EQ(walnuts::detail::log_sum_exp(x), walnuts::detail::log_sum_exp(1.0, 2.0));
}

TEST(LogSumExpVector, MixedInfinities) {
  double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x(3);
  x << inf, 1.0, -inf;
  auto lse = walnuts::detail::log_sum_exp(x);
  EXPECT_TRUE(std::isinf(lse) && lse > 0);
}

TEST(LogSumExpVector, NaNPropagates) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd x(3);
  x << 1.0, nan, 2.0;
  EXPECT_TRUE(std::isnan(walnuts::detail::log_sum_exp(x)));
}

// walnuts::detail::logp_momentum() function *****************************************

TEST(LogpMomentum, SingleElement) {
  Eigen::VectorXd rho(1), inv_mass(1);
  rho << 2.0;
  inv_mass << 1.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, inv_mass), -2.0);
}

TEST(LogpMomentum, ZeroMomentum) {
  Eigen::VectorXd rho = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd inv_mass = Eigen::VectorXd::Constant(4, 2.5);
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, inv_mass), 0.0);
}

TEST(LogpMomentum, UnitMass3Vector) {
  Eigen::VectorXd rho(3), inv_mass(3);
  rho << 1.0, 2.0, 3.0;
  inv_mass << 1.0, 1.0, 1.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, inv_mass), -0.5 * (1.0 + 4.0 + 9.0));
}

TEST(LogpMomentum, KnownValueLinearTransform) {
  Eigen::VectorXd rho(2), inv_mass(2);
  rho << 2.0, 3.0;
  inv_mass << 0.5, 2.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, inv_mass), -10.0);
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, inv_mass),
                   walnuts::detail::logp_momentum(-rho, inv_mass));
  EXPECT_DOUBLE_EQ(walnuts::detail::logp_momentum(rho, 2.5 * inv_mass),
		   2.5 * walnuts::detail::logp_momentum(rho, inv_mass));
}

// NoExceptLogpGrad class *******************************************

struct ThrowingLogpGrad {
  void operator()(const Eigen::VectorXd& x, double& logp,
                  Eigen::VectorXd& grad) const {
    throw std::runtime_error("logp_grad failed");
  }
};

struct GoodLogpGrad {
  void operator()(const Eigen::VectorXd& x, double& logp,
                  Eigen::VectorXd& grad) const {
    logp = -0.5 * x.squaredNorm();
    grad = -x;
  }
};

TEST(NoExceptLogpGrad, NormalEvaluation) {
  GoodLogpGrad f;
  walnuts::detail::NoExceptLogpGrad wrapped(f);
  Eigen::VectorXd x(3);
  x << 1.0, 2.0, 3.0;
  double logp;
  Eigen::VectorXd grad;
  wrapped(x, logp, grad);
  EXPECT_DOUBLE_EQ(logp, -0.5 * x.squaredNorm());
  expect_near(grad, (-x).eval());
}

TEST(NoExceptLogpGrad, ExceptionSetsNegInfAndZeroGrad) {
  ThrowingLogpGrad f;
  walnuts::detail::NoExceptLogpGrad wrapped(f);
  Eigen::VectorXd x(3);
  x << 1.0, 2.0, 3.0;
  double logp = 0.0;
  Eigen::VectorXd grad = Eigen::VectorXd::Ones(3);
  wrapped(x, logp, grad);
  EXPECT_EQ(logp, -std::numeric_limits<double>::infinity());
  expect_near(grad, Eigen::VectorXd::Zero(3).eval());
}

// grad() function **************************************************

TEST(DetailGrad, ReturnsCorrectGradient) {
  GoodLogpGrad f;
  Eigen::VectorXd theta(3);
  theta << 1.0, 2.0, 3.0;
  Eigen::VectorXd g = walnuts::detail::grad(f, theta);
  expect_near(g, (-theta).eval());
}

TEST(DetailGrad, ZeroAtOrigin) {
  GoodLogpGrad f;
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd g = walnuts::detail::grad(f, theta);
  expect_near(g, Eigen::VectorXd::Zero(4).eval());
}

// l2_rel_diff() function *******************************************

TEST(DetailL2RelDiff, SingleElement) {
  Eigen::VectorXd a(1), b(1);
  a << 2.0;
  b << 1.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::l2_rel_diff(a, b), 1.0);  // (2-1)/1 = 1
}

TEST(DetailL2RelDiff, ZeroDifference) {
  Eigen::VectorXd a(3);
  a << 1.0, 2.0, 3.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::l2_rel_diff(a, a), 0.0);
}

TEST(DetailL2RelDiff, KnownValue) {
  Eigen::VectorXd a(2), b(2);
  a << 3.0, 5.0;
  b << 1.0, 4.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::l2_rel_diff(a, b), std::sqrt(2.0*2.0 + 0.25*0.25));
}

TEST(DetailL2RelDiff, ScaleInvariant) {
  Eigen::VectorXd a(3), b(3);
  a << 2.0, 3.0, 4.0;
  b << 1.0, 2.0, 3.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::l2_rel_diff(3.0 * a, 3.0 * b),
		   walnuts::detail::l2_rel_diff(a, b));
}

// sum() function ***************************************************

TEST(DetailSum, KnownValues) {
  EXPECT_EQ(walnuts::detail::sum(std::vector<std::size_t>{}), 0UL);
  EXPECT_EQ(walnuts::detail::sum(std::vector<std::size_t>{7}), 7UL);
  EXPECT_EQ(walnuts::detail::sum(std::vector<std::size_t>{0, 1, 2, 3, 4, 5}), 15UL);
}

// variance() function **********************************************

TEST(DetailVariance, KnownValue) {
  Eigen::VectorXd xs(4);
  xs << 2.0, 4.0, 4.0, 8.0;
  double m = (2.0 + 4.0 + 4.0 + 8.0) / 4.0;
  double sq_diffs = (2.0 - m) * (2.0 - m)
    + (4.0 - m) * (4.0 - m)
    + (4.0 - m) * (4.0 - m)
    + (8.0 - m) * (8.0 - m);
  EXPECT_DOUBLE_EQ(walnuts::detail::variance(xs), sq_diffs / 3.0);
}

TEST(DetailVariance, ConstantVector) {
  Eigen::VectorXd xs = Eigen::VectorXd::Constant(5, 3.7);
  EXPECT_DOUBLE_EQ(walnuts::detail::variance(xs), 0.0);
}

TEST(DetailVariance, ShiftInvariantQuadraticScale) {
  Eigen::VectorXd xs(4);
  xs << 1.0, 2.0, 3.0, 4.0;
  EXPECT_DOUBLE_EQ(walnuts::detail::variance(xs + Eigen::VectorXd::Constant(4, 100.0)),
                   walnuts::detail::variance(xs));
  EXPECT_DOUBLE_EQ(walnuts::detail::variance(2.0 * xs),
                   4.0 * walnuts::detail::variance(xs));
  EXPECT_DOUBLE_EQ(walnuts::detail::variance((5.7 + (13.2 * xs).array()).matrix()),
                   13.2 * 13.2 * walnuts::detail::variance(xs));
}
