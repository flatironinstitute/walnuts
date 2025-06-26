#include <chrono>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <walnuts/online_moments.hpp>

static Eigen::VectorXd discounted_mean(const std::vector<Eigen::VectorXd>& ys,
                                double alpha) {
  std::size_t N = ys.size();
  long D = ys[0].size();
  double weight_sum = 0;
  Eigen::VectorXd weighted_value_sum = Eigen::VectorXd::Zero(D);
  for (std::size_t n = 0; n < N; ++n) {
    double weight = std::pow(alpha, N - n - 1);
    weight_sum += weight;
    weighted_value_sum += weight * ys[n];
  }
  return weighted_value_sum / weight_sum;
}

static Eigen::VectorXd discounted_variance(const std::vector<Eigen::VectorXd>& ys,
                                    double alpha) {
  Eigen::VectorXd mu = discounted_mean(ys, alpha);
  std::size_t N = ys.size();
  long D = ys[0].size();
  double weight_sum = 0;
  Eigen::VectorXd weighted_sq_diff_sum = Eigen::VectorXd::Zero(D);
  for (std::size_t n = 0; n < N; ++n) {
    double weight = std::pow(alpha, N - n - 1);
    weight_sum += weight;
    auto diff = (ys[n] - mu).array();
    weighted_sq_diff_sum += weight * (diff * diff).matrix();
  }
  return weighted_sq_diff_sum / weight_sum;
}

TEST(Welford, test_zero_observations) {
  double alpha = 0.95;
  long D = 2;
  nuts::OnlineMoments<double> acc(alpha, D);

  Eigen::VectorXd m = acc.mean();
  Eigen::VectorXd v = acc.variance();
  EXPECT_EQ(2, m.size());
  EXPECT_FLOAT_EQ(0.0, m(0));
  EXPECT_FLOAT_EQ(0.0, m(1));
  EXPECT_EQ(2, v.size());
  EXPECT_FLOAT_EQ(0.0, v(0));
  EXPECT_FLOAT_EQ(0.0, v(1));
}

TEST(Welford, test_one_observation) {
  double alpha = 0.95;
  long D = 2;
  nuts::OnlineMoments<double> acc(alpha, D);

  Eigen::VectorXd y(2);
  y << 0.2, -1.3;
  acc.observe(y);

  Eigen::VectorXd m = acc.mean();
  Eigen::VectorXd v = acc.variance();

  EXPECT_EQ(2, m.size());
  EXPECT_FLOAT_EQ(0.2, m(0));
  EXPECT_FLOAT_EQ(-1.3, m(1));
  EXPECT_EQ(2, v.size());
  EXPECT_FLOAT_EQ(0.0, v(0));
  EXPECT_FLOAT_EQ(0.0, v(1));
}

TEST(Welford, test_no_discounting) {
  long D = 2;
  std::size_t N = 100;
  std::vector<Eigen::VectorXd> ys(N);
  for (std::size_t n = 0; n < N; ++n) {
    ys[n] = Eigen::VectorXd::Zero(D);
  }
  for (std::size_t n = 0; n < N; ++n) {
    double x = static_cast<double>(n);
    ys[n] << x, std::sqrt(x);
  }

  Eigen::VectorXd sum = Eigen::VectorXd::Zero(D);
  for (auto y : ys) {
    sum += y;
  }
  Eigen::VectorXd mean_expected = sum / N;

  Eigen::VectorXd sum_sq_diffs = Eigen::VectorXd::Zero(D);
  for (auto y : ys) {
    sum_sq_diffs +=
        ((y - mean_expected).array() * (y - mean_expected).array()).matrix();
  }
  Eigen::VectorXd variance_expected = sum_sq_diffs / N;

  double alpha = 1.0;
  nuts::OnlineMoments<double> acc(alpha, D);

  for (std::size_t n = 0; n < N; ++n) {
    acc.observe(ys[n]);
  }
  Eigen::VectorXd m = acc.mean();
  Eigen::VectorXd v = acc.variance();

  EXPECT_TRUE(m.isApprox(mean_expected, 1e-8));
  EXPECT_TRUE(v.isApprox(variance_expected, 1e-8));
}

TEST(Welford, test_ten_observations) {
  long D = 3;
  std::size_t N = 10;
  std::vector<Eigen::VectorXd> ys(N);
  for (std::size_t n = 0; n < N; ++n) {
    ys[n] = Eigen::VectorXd::Zero(D);
  }
  for (std::size_t n = 0; n < N; ++n) {
    double x = static_cast<double>(n);
    ys[n] << x, x * x, std::exp(x);
  }

  double alpha = 0.95;
  nuts::OnlineMoments<double> acc(alpha, D);

  for (std::size_t n = 0; n < N; ++n) {
    acc.observe(ys[n]);
  }
  Eigen::VectorXd m = acc.mean();
  Eigen::VectorXd v = acc.variance();

  Eigen::VectorXd mean_expected = discounted_mean(ys, alpha);
  EXPECT_TRUE(m.isApprox(mean_expected, 1e-8));

  Eigen::VectorXd variance_expected = discounted_variance(ys, alpha);
  EXPECT_TRUE(v.isApprox(variance_expected, 1e-8));
}
