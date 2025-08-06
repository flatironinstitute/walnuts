#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <walnuts/adaptive_walnuts.hpp>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;
using Integer = int64_t;

static void normal_logp_grad(const VectorS& x, S& logp,
			     VectorS& grad) {
  const auto D = x.size();
  grad = VectorS::Zero(D);
  logp = 0;
  for (auto d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
}

int main() {
  Integer D = 20;
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 0.1;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);

  double step_size_init = 0.5;
  double accept_rate_target = 2.0 / 3.0;
  double step_iteration_offset = 2.0;
  double learning_rate = 0.95;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);

  S max_error = 1.0;  // 61% Metropolis
  Integer max_nuts_depth = 10;
  Integer max_step_depth = 8;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);

  unsigned int seed = 428763;
  std::mt19937 rng(seed);

  std::normal_distribution<S> std_normal(0, 1);
  VectorS theta_init(D);
  for (Integer i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  std::cout << "\nADAPTIVE WALNUTS" << std::endl;
  nuts::AdaptiveWalnuts walnuts(rng, normal_logp_grad, theta_init, mass_cfg,
                                step_cfg, walnuts_cfg);

  Integer warmup_iterations = 1000;
  for (Integer n = 0; n < warmup_iterations; ++n) {
    walnuts();
  }

  auto file_name = "walnuts-stationarity.csv";
  std::ofstream out(file_name);
  out << std::fixed << std::setprecision(8);
  Integer sampling_iterations = 10000000;
  auto sampler = walnuts.sampler();
  for (Integer n = 0; n < sampling_iterations; ++n) {
    if (n % 1000000 == 0) {
      std::cout << "   iteration: " << n
		<< "/" << sampling_iterations << std::endl;
    }
    VectorS theta = sampler();
    for (int d = 0; d < D; ++d) {
      if (d > 0) {
	out << ',';
      }
      out << theta(d);
    }
    out << '\n';
  }
  out.close();
  std::cout << "WROTE " << sampling_iterations << " " << D
	    << "-DIMENSIONAL DRAWS TO build/walnuts-convergence.csv" << std::endl;
  return 0;
}
