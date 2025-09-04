#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include "load_stan.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
		      const std::vector<double> lps,
		      const std::string& filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file " + filename);
  }
  out << std::setprecision(12);
  out << "lp_grad_calls,logp";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out << ',' << names[i];
  }
  out << '\n';
  for (int col = 0; col < draws.cols(); ++col) {
    out << lp_grads[static_cast<std::size_t>(col)]
	<< ',' << lps[static_cast<std::size_t>(col)];
    for (int row = 0; row < draws.rows(); ++row) {
      out << ',' << draws(row, col);
    }
    out << '\n';
  }
}


template <typename RNG>
static VectorS std_normal(int N, RNG& rng) {
  std::normal_distribution<S> norm(0.0, 1.0);
  VectorS y(N);
  for (Integer n = 0; n < N; ++n) {
    y(n) = norm(rng);
  }
  return y;
}

static void test_adaptive_walnuts(const DynamicStanModel& model,
				  const std::string& sample_csv_file,
				  Integer seed,
				  Integer iter_warmup,
				  Integer iter_sampling) {
  int D = model.unconstrained_dimensions();

  std::mt19937 rng(static_cast<unsigned int>(seed));  // TODO: use unsigned long

  long logp_grad_calls = 0;
  auto logp = [&](auto&&... args) {
    ++logp_grad_calls;
    model.logp_grad(args...);
  };

  VectorS theta_init = std_normal(D, rng);

  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 1e-5;  // max init variance is 1 / additive_smoothing
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);
  
  double step_size_init = 0.1;
  double accept_rate_target = 0.8;  // min 2.0 / 3.0
  double step_iteration_offset = 2.0;
  double learning_rate = 2.0;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);

  double max_error = 0.5;  // 1000: NUTS; 1.0: 37% accept; 0.5: 62%; 0.2: 82%
  Integer max_nuts_depth = 8;
  Integer max_step_depth = 6;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);

  nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init,
				mass_cfg, step_cfg, walnuts_cfg);

  for (Integer n = 0; n < iter_warmup; ++n) {
    walnuts();
  }

  auto sampler = walnuts.sampler();
  int M = model.constrained_dimensions();
  MatrixS draws(M, iter_sampling);
  std::vector<Integer> lp_grads(static_cast<std::size_t>(iter_sampling));
  std::vector<double> lps(lp_grads.size());
  double lp = 0.0;
  Eigen::VectorXd grad_dummy(D);
  Eigen::VectorXd constrained_draw(M);
  for (Integer n = 0; n < iter_sampling; ++n) {
    auto draw = sampler();
    model.constrain_draw(draw, draws.col(n));
    lp_grads[static_cast<std::size_t>(n)] = logp_grad_calls;
    model.logp_grad(draw, lp, grad_dummy);  // redundant, but can't easily recover
    lps[static_cast<std::size_t>(n)] = lp;
  }
  
  write_csv(model.param_names(), draws, lp_grads, lps, sample_csv_file);
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: "
	      << argv[0]
	      << " <dir> <model> <seed> <iter_warmup> <iter_sampling> <trials>"
	      << std::endl;
    return 1;
  }
  std::string dir = argv[1];
  std::string model = argv[2];
  unsigned int seed = static_cast<unsigned int>(std::stoul(argv[3]));
  int iter_warmup = std::stoi(argv[4]);
  int iter_sampling = std::stoi(argv[5]);
  int trials = std::stoi(argv[6]);
  std::string prefix = dir + "/" + model + "/" + model;
  std::string model_so_file = prefix + "_model.so";
  std::string data_json_file = prefix + "-data.json";
  std::string sample_csv_file = prefix + "-walnuts-draws.csv";

  std::cout << "model= " << model << std::endl;
  std::cout << "seed= " << seed << std::endl;
  std::cout << "iter_warmup= " << iter_warmup << std::endl;
  std::cout << "iter_sampling= " << iter_sampling << std::endl;

  auto model_so_file_c_str = model_so_file.c_str();
  auto data_json_file_c_str = data_json_file.c_str();
  DynamicStanModel stan_model(model_so_file_c_str, data_json_file_c_str, seed);
  for (int trial = 0; trial < trials; ++trial) {
    std::cout << "trial = " << trial << std::endl;
    unsigned int trial_seed = seed + static_cast<unsigned int>(17 * (trial + 1));
    std::string sample_csv_file_numbered = prefix + "-walnuts-draws-" + std::to_string(trial) + ".csv";
    test_adaptive_walnuts(stan_model, sample_csv_file_numbered, trial_seed, iter_warmup, iter_sampling);
  }
  std::quick_exit(0);  // crashes without this---not stan_model dtor, prob dlclose_deleter
}
