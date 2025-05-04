#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <chrono>
#include "nuts.hpp"

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

double total_time = 0.0;
int count = 0;

template <typename T>
void standard_normal_logp_grad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                                S& logp,
                                Eigen::Matrix<T, Eigen::Dynamic, 1>& grad) {
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
  S step_size = 0.025;
  int max_depth = 10;
  std::cout << "D = " << D << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << std::endl;
  MatrixS draws(D, N);

  std::mt19937 rng(init_seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  VectorS inv_mass = VectorS::Ones(D);

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts::nuts<S>(seed, standard_normal_logp_grad<S>, inv_mass, step_size, max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time = std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "total time: " << global_total_time << "s" << std::endl;
  std::cout << "    gradient time: " << total_time << "s" << std::endl;
  std::cout << "        gradient calls: " << count << std::endl;
  std::cout << "        gradient time per call: " << total_time / count << "s" << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < D; ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev << "\n";
  }

  return 0;
}
