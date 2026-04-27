#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace nuts {

  Eigen::VectorXd zero_validate(const std::vector<Eigen::VectorXd>& xs) {
    if (xs.size() == 0u) return {0};
    int64_t D = xs[0].size();
    for (const auto& x : xs) {
      if (x.size() != D) {
	throw std::invalid_argument("all vectors must be the same size");
      }
    }
    return {D};
  }

  Eigen::VectorXd sum(const std::vector<Eigen::VectorXd>& xs) {
    Eigen::VectorXd sum = zero_validate(xs);
    for (const auto& x : xs) {
      sum += x.sum();
    }
    return sum;
  }

  Eigen::VectorXd sum_squares(const std::vector<Eigen::VectorXd>& xs) {
    Eigen::VectorXd sum = zero_validate(xs);
    for (const auto& x : xs) {
      sum += x.array().square().sum();
    }
    return sum;
  }

  Eigen::VectorXd mean(const std::vector<Eigen::VectorXd>& xs) {
    return sum(xs) / xs.size();
  }

}
