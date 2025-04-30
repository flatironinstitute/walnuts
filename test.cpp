#include "nuts.hpp"
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>

template <typename S>
void standard_normal_logp_grad(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                                S& logp,
                                Eigen::Matrix<S, Eigen::Dynamic, 1>& grad) {
  logp = -0.5 * x.dot(x);
  grad = -x;
}

int main() {
  constexpr int D = 10;
  constexpr int N = 100000;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> draws(N, D);

  std::random_device rd;
  std::mt19937 rng(rd());

  Eigen::VectorXd theta_init(D);
  std::normal_distribution<> std_normal(0.0, 1.0);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);

  double step_size = 0.1;
  int max_depth = 10;
  nuts<double>(rng, standard_normal_logp_grad<double>, inv_mass, step_size, max_depth, theta_init, draws);

  for (int d = 0; d < D; ++d) {
    double mean = draws.col(d).mean();
    double var = (draws.col(d).array() - mean).square().sum() / (N - 1);
    double stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev << "\n";
  }

  return 0;
}
