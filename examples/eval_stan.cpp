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
		      const std::vector<Integer> lp_grads,
		      const std::string& filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file " + filename);
  }
  out << "lp_grad_calls";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out << ',' << names[i];
  }
  out << '\n';

  for (int col = 0; col < draws.cols(); ++col) {
    out << lp_grads[static_cast<std::size_t>(col)];
    for (int row = 0; row < draws.rows(); ++row) {
      out << ',' << draws(row, col);
    }
    out << '\n';
  }
}


template <typename RNG>
static VectorS std_normal(int N, RNG rng) {
  std::normal_distribution<S> norm(0.0, 1.0);
  VectorS y(N);
  for (Integer n = 0; n < N; ++n) {
    y(n) = norm(rng);
  }
  return y;
}

static void test_adaptive_walnuts(const DynamicStanModel& model, unsigned int seed,
				  const char* out_csv) {
  int D = model.unconstrained_dimensions();
  std::mt19937 rng(seed);

  long logp_grad_calls = 0;
  auto logp = [&](auto&&... args) {
    ++logp_grad_calls;
    model.logp_grad(args...);
  };

  VectorS theta_init = std_normal(D, rng);

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

  double max_error = 5.0;  // 1.0: 37% accept; 0.5: 62%; 0.2: 82%
  Integer max_nuts_depth = 10;
  Integer max_step_depth = 6;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);

  nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init,
				mass_cfg, step_cfg, walnuts_cfg);

  Integer iter_warmup = 64;
  Integer iter_sampling = 5000;

  for (Integer n = 0; n < iter_warmup; ++n) {
    walnuts();
  }

  auto sampler = walnuts.sampler();
  int M = model.constrained_dimensions();
  MatrixS draws(M, iter_sampling);
  std::vector<Integer> lp_grads(static_cast<std::size_t>(iter_sampling));
  for (Integer n = 0; n < iter_sampling; ++n) {
    model.constrain_draw(sampler(), draws.col(n));
    lp_grads[static_cast<std::size_t>(n)] = logp_grad_calls;
  }

  write_csv(model.param_names(), draws, lp_grads, out_csv);
}

int main(int argc, char** argv) {
  char* lib{nullptr};
  char* data{nullptr};
  char* out_csv{nullptr};
  if (argc <= 1 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.so> [data.json] <out.csv>" << std::endl;
    return 1;
  }
  if (argc == 3) {
    lib = argv[1];
    out_csv = argv[2];
  }
  if (argc == 4) {
    lib = argv[1];
    data = argv[2];
    out_csv = argv[3];
  }

  unsigned int seed = 428763;
  DynamicStanModel model(lib, data, seed);
  test_adaptive_walnuts(model, seed, out_csv);
  
  return 0;
}
