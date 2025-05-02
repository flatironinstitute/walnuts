#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <chrono>
#include "nuts.hpp"

double total_time = 0.0;
int count = 0;

template <typename S>
void standard_normal_logp_grad(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                                S& logp,
                                Eigen::Matrix<S, Eigen::Dynamic, 1>& grad) {
  auto start = std::chrono::high_resolution_clock::now();
  logp = -0.5 * x.dot(x);
  grad = -x;
  auto end = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end - start).count();
  ++count;
}

int main() {
  int init_seed = 333456;
  int seed = 763545;
  int D = 10;
  int N = 10000;
  double step_size = 0.025;
  int max_depth = 10;
  std::cout << "D = " << D << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << std::endl;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> draws(D, N);

  std::mt19937 rng(init_seed);
  std::normal_distribution<> std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts<double>(seed, standard_normal_logp_grad<double>, inv_mass, step_size, max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time = std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "total time: " << global_total_time << "s" << std::endl;
  std::cout << "    gradient time: " << total_time << "s" << std::endl;
  std::cout << "        gradient calls: " << count << std::endl;
  std::cout << "        gradient time per call: " << total_time / count << "s" << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < D; ++d) {
    double mean = draws.row(d).mean();
    double var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    double stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev << "\n";
  }

  return 0;
}
