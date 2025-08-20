#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include "load_stan.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;
using Integer = long;

static void write_csv(const std::vector<std::string>& names,
		      const Eigen::MatrixXd& draws,
		      const std::string& filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file " + filename);
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << ',';
    out << names[i];
  }
  out << "\n";
  for (int col = 0; col < draws.cols(); ++col) {
    for (int row = 0; row < draws.rows(); ++row) {
      if (row > 0) out << ',';
      out << draws(row, col);
    }
    out << "\n";
  }
}


static void test_adaptive_walnuts(const DynamicStanModel& model, unsigned int seed) {
  int D = model.unconstrained_dimensions();
  std::mt19937 rng(seed);
  Integer iter_warmup = 64;
  Integer iter_sampling = 5000;
  Integer max_nuts_depth = 10;
  S max_error = 5.0;  // 1.0: 37% accept; 0.5: min 62% accept;  0.2: min 82% accept

  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (Integer i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 0.1;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);
  double step_size_init = 0.5;
  double accept_rate_target = 0.8;  // minimum 2.0 / 3.0;
  double step_iteration_offset = 2.0;
  double learning_rate = 0.95;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);
  Integer max_step_depth = 6;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);
  long logp_grad_calls = 0;
  auto logp = [&](auto&&... args) {
    ++logp_grad_calls;
    model.logp_grad(args...);
  };
  nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init, mass_cfg, step_cfg,
                                walnuts_cfg);

  for (Integer n = 0; n < iter_warmup; ++n) {
    walnuts();
    std::cout << "warmup( " << n << "): logp_grad_calls = " << logp_grad_calls << std::endl;
  }

  // N post-warmup draws
  auto sampler = walnuts.sampler();  // freeze tuning
  std::cout << "Adaptation completed." << std::endl;
  std::cout << "Macro step size = " << sampler.macro_step_size() << std::endl;
  std::cout << "Max error = " << sampler.max_error() << std::endl;

  int M = model.constrained_dimensions();
  MatrixS draws(M, iter_sampling);
  for (Integer n = 0; n < iter_sampling; ++n) {
    model.constrain_draw(sampler(), draws.col(n));
    std::cout << "sampling(" << n << "): logp_grad_calls = " << logp_grad_calls << std::endl;
  }
  write_csv(model.param_names(), draws, "out.csv");
}

int main(int argc, char** argv) {
  char* lib{nullptr};
  char* data{nullptr};
  if (argc <= 1 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <model_path.so> [data]" << std::endl;
    return 1;
  }
  lib = argv[1];
  if (argc == 3) {
    data = argv[2];
  }

  unsigned int seed = 428763;
  DynamicStanModel model(lib, data, seed);
  test_adaptive_walnuts(model, seed);
  
  return 0;
}
