#pragma once

#include <limits>
#include <vector>

#include <Eigen/Dense>

static std::vector<double> inf_nan() {
  std::vector<double> result;
  result.push_back(std::numeric_limits<double>::quiet_NaN());
  result.push_back(std::numeric_limits<double>::infinity());
  return result;
}

static std::vector<double> inf_nan_neg() {
  std::vector<double> result = inf_nan();
  result.push_back(-0.3);
  return result;
}

static std::vector<double> inf_nan_neg_zero() {
  std::vector<double> result = inf_nan_neg();
  result.push_back(0.0);
  return result;
}

static std::vector<double> inf_nan_neg_zero_geq_one() {
  std::vector<double> result = inf_nan_neg_zero();
  result.push_back(1.0);
  result.push_back(1.5);
  return result;
}

static std::vector<double> inf_nan_neg_zero_leq_one() {
  std::vector<double> result = inf_nan_neg_zero();
  result.push_back(1.0);
  result.push_back(0.99);
  return result;
}

template <int R, int C>
static void expect_near(const Eigen::Matrix<double, R, C>& x,
			const Eigen::Matrix<double, R, C>& y,
			double tolerance = 1e-10) {
  EXPECT_EQ(x.size(), y.size());
  for (Eigen::Index n = 0; n < x.size(); ++n) {
    EXPECT_NEAR(x(n), y(n), tolerance);
  }
}

static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}
