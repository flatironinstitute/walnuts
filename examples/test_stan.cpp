#include <walnuts/adaptive_walnuts.hpp>
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
using Integer = long;

static void summarize(const MatrixS &draws) {
  Integer N = draws.cols();
  Integer D = draws.rows();
  for (Integer d = 0; d < D; ++d) {
    if (d > 3 && d < D - 3) {
      if (d == 4) {
        std::cout << "... elided " << (D - 6) << " rows ..." << std::endl;
      }
      continue;
    }
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
}

template <typename RNG>
static void test_nuts(const DynamicStanModel &model, const VectorS &theta_init,
                      RNG &rng, Integer D, Integer N, S step_size,
                      Integer max_depth, const VectorS &inv_mass) {
  std::cout << "\nTEST NUTS" << std::endl;
  model.total_time_ = 0.0;
  model.count_ = 0;

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts::Random<double, RNG> rand(rng);

  auto logp = [&model](auto &&...args) { model.logp_grad(args...); };

  nuts::Nuts sample(rand, logp, theta_init, inv_mass, step_size, max_depth);
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    model.constrain_draw(sample(), draws.col(n));
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

  summarize(draws);
}

template <typename RNG>
static void test_walnuts(const DynamicStanModel &model, VectorS theta_init,
                         RNG &rng, Integer D, Integer N, S macro_step_size,
                         Integer max_nuts_depth, S log_max_error,
                         VectorS inv_mass) {
  std::cout << "\nTEST WALNUTS" << std::endl;
  nuts::Random<double, RNG> rand(rng);
  auto logp = [&model](auto &&...args) { model.logp_grad(args...); };

  nuts::WalnutsSampler sample(rand, logp, theta_init, inv_mass, macro_step_size,
                              max_nuts_depth, log_max_error);
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    model.constrain_draw(sample(), draws.col(n));
  }
  summarize(draws);
}

template <typename RNG>
static void test_adaptive_walnuts(const DynamicStanModel &model,
                                  const VectorS &theta_init, RNG &rng,
                                  Integer D, Integer N, Integer max_nuts_depth,
                                  S log_max_error) {
  std::cout << "\nTEST ADAPTIVE WALNUTS" << std::endl;
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 20.0;
  double mass_iteration_offset = 20.0;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset);

  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_iteration_offset = 4.0;
  double learning_rate = 0.95;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);

  Integer max_step_depth = 8;
  nuts::WalnutsConfig walnuts_cfg(log_max_error, max_nuts_depth,
                                  max_step_depth);
  auto logp = [&model](auto &&...args) { model.logp_grad(args...); };

  nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init, mass_cfg, step_cfg,
                                walnuts_cfg);

  for (Integer n = 0; n < N; ++n) {
    walnuts();
  }

  // N post-warmup draws
  auto sample = walnuts.sampler();  // freeze tuning
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    model.constrain_draw(sample(), draws.col(n));
  }
  summarize(draws);
}

int main(int argc, char **argv) {
  unsigned int seed = 428763;
  Integer N = 2000;
  S step_size = 0.5;
  Integer max_depth = 10;
  S log_max_error = 0.1;  // 80% Metropolis, 45% Barker

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
  VectorS inv_mass = VectorS::Ones(D);

  std::cout << "SHARED CONSTANTS:" << std::endl;
  std::cout << "D = " << D << ";  N = " << N << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth
            << ";  log_max_error = " << log_max_error << std::endl;

  std::mt19937 rng(seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (Integer i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  test_nuts(model, theta_init, rng, D, N, step_size, max_depth, inv_mass);

  test_walnuts(model, theta_init, rng, D, N, step_size, max_depth,
               log_max_error, inv_mass);

  test_adaptive_walnuts(model, theta_init, rng, D, N, max_depth, log_max_error);

  return 0;
}
