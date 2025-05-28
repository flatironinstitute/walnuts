#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include "load_stan.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

enum class Sampler { Nuts, Walnuts };

template <Sampler U, typename G>
void test_nuts(const DynamicStanModel &model, const VectorS &theta_init,
               G &generator, int N, S step_size, S max_depth, S max_error,
               const VectorS &inv_mass) {
  model.total_time_ = 0.0;
  model.count_ = 0;
  std::cout << std::endl
            << "D = " << model.unconstrained_dimensions() << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << ";  WALNUTS = " << (U == Sampler::Walnuts ? "true" : "false")
            << std::endl;

  auto logp = [&model](auto &&...args) { model.logp_grad(args...); };

  int M = model.constrained_dimensions();

  MatrixS draws(M, N);
  auto writer = [&model, &draws](int n, const VectorS &theta) {
    model.constrain_draw(theta, draws.col(n));
  };

  auto global_start = std::chrono::high_resolution_clock::now();
  if constexpr (U == Sampler::Walnuts) {
    nuts::walnuts(generator, logp, inv_mass, step_size, max_depth, max_error,
                     theta_init, N, writer);
  } else if constexpr (U == Sampler::Nuts) {
    nuts::nuts(generator, logp, inv_mass, step_size, max_depth, theta_init, N,
               writer);
  }
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time =
      std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "    total time: " << global_total_time << "s" << std::endl;
  std::cout << "logp_grad time: " << model.total_time_ << "s" << std::endl;
  std::cout << "logp_grad fraction: " << model.total_time_ / global_total_time
            << std::endl;
  std::cout << "        logp_grad calls: " << model.count_ << std::endl;
  std::cout << "        time per call: " << model.total_time_ / model.count_
            << "s" << std::endl;
  std::cout << std::endl;

  auto names = model.param_names();
  for (int d = 0; d < std::min(M, 5); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << names[d] << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (M > 5) {
    std::cout << "... elided " << (M - 5) << " dimensions ..." << std::endl;
  }
}

int main(int argc, char *argv[]) {
  int seed = 333456;
  int N = 5000;
  double step_size = 0.025;
  int max_depth = 10;
  double max_error = 0.2;  // 80% Metropolis, 45% Barker

  char *lib;
  char *data;

  if (argc <= 1) {
    // require at least the library name
    std::cerr << "Usage: " << argv[0] << " <model_path> [data]" << std::endl;
    return 1;
  } else if (argc == 2) {
    lib = argv[1];
    data = NULL;
  } else {
    lib = argv[1];
    data = argv[2];
  }

  DynamicStanModel model(lib, data, seed);

  int D = model.unconstrained_dimensions();

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);
  std::mt19937 generator(seed);
  std::normal_distribution<double> std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  test_nuts<Sampler::Nuts>(model, theta_init, generator, N, step_size,
                           max_depth, max_error, inv_mass);
  test_nuts<Sampler::Walnuts>(model, theta_init, generator, N, step_size,
                              max_depth, max_error, inv_mass);

  return 0;
}
