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
constexpr std::array y_obs{1,1,0,0,0,0,0,0,0,0};
template <typename T>
void bernoulli_logp_grad(const nuts::Vec<T>& x,
                         double& logp,
                         nuts::Vec<T>& grad) {
  auto start = std::chrono::high_resolution_clock::now();
  const int D = x.size();
  logp = 0.0;
  grad.setZero();
  for (int i = 0; i < D; ++i) {
    double xi = x(i);
    double pi = 1.0 / (1.0 + std::exp(-xi));       // inv_logit
    // likelihood
    for (int j = 0; j < y_obs.size(); ++j) {
      int y = y_obs[j];
      logp += y *  std::log(pi)
            + (1-y)*std::log(1.0 - pi);
    }
    // Jacobian of inv_logit: log |d p / d x| = log(pi) + log(1-pi)
    logp += std::log(pi) + std::log(1.0 - pi);

    // gradient w.r.t. x_i:
    //   ∂ℓ/∂p = ∑[ y/p  − (1−y)/(1−p) ]
    //   ∂p/∂x = p(1−p)
    double dlogp_dp = 0;
    for (int j = 0; j < y_obs.size(); ++j) {
      int y = y_obs[j];
      dlogp_dp +=  y/pi  -  (1-y)/(1.0 - pi);
    }
    // plus Jacobian derivative term = (1-2p)
    double dlogjac_dx = 1.0 - 2.0*pi;
    grad(i) = dlogp_dp * (pi*(1.0 - pi))   // chain‐rule
              + dlogjac_dx;
  }
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
  VectorS inv_mass = VectorS::Ones(D);
  MatrixS draws(D, N);
  std::cout << "D = " << D << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << std::endl;

  std::mt19937 generator(init_seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }


  auto global_start = std::chrono::high_resolution_clock::now();
  nuts::nuts(generator, bernoulli_logp_grad<S>, inv_mass, step_size, max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time = std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "    total time: " << global_total_time << "s" << std::endl;
  std::cout << "logp_grad time: " << total_time << "s" << std::endl;
  std::cout << "logp_grad fraction: " << total_time / global_total_time << std::endl;
  std::cout << "        logp_grad calls: " << count << std::endl;
  std::cout << "        time per call: " << total_time / count << "s" << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < D; ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev << "\n";
  }
  return 0;
}
