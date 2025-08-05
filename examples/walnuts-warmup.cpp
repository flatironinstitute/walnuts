#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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
  Integer D = 200;
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 1e-12; 
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);

  double step_size_init = 1;
  double accept_rate_target = 2.0 / 3.0; // 2.0 / 3.0;
  double step_iteration_offset = 1.1;  // stan default: 10.0
  double learning_rate = 0.8;  // stan default: 0.75
  double decay_rate = 0.05;  // stan default: 0.05
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);

  S max_error = 1.0;  // 61% Metropolis
  Integer max_nuts_depth = 10;
  Integer max_step_depth = 8;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);

  unsigned int seed = 8472222;
  std::mt19937 rng(seed);

  std::normal_distribution<S> std_normal(0, 1);
  VectorS theta_init(D);
  for (Integer i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  std::cout << "\nADAPTIVE WALNUTS" << std::endl;
  nuts::AdaptiveWalnuts walnuts(rng, normal_logp_grad, theta_init, mass_cfg,
                                step_cfg, walnuts_cfg);

  auto file_name_inv_mass = "walnuts-warmup-inv-mass.csv";
  std::ofstream out_inv_mass(file_name_inv_mass);
  out_inv_mass << std::fixed << std::setprecision(8);

  auto file_name_step = "walnuts-warmup-step.csv";
  std::ofstream out_step(file_name_step);
  out_step << std::fixed << std::setprecision(8);
  
  Integer warmup_iterations = 1000;
  for (Integer n = 0; n < warmup_iterations; ++n) {
    walnuts();

    VectorS inv_mass_diag = walnuts.inv_mass();
    for (int d = 0; d < D; ++d) {
      if (d > 0) {
	out_inv_mass << ',';
      }
      out_inv_mass << inv_mass_diag(d);
    }
    out_inv_mass << '\n';

    S step_size = walnuts.step_size();
    out_step << step_size << "\n";
  }
  out_inv_mass.close();
  out_step.close();

  std::cout << "WROTE " << warmup_iterations << " " << D
	    << "-DIMENSIONAL DRAWS TO build/walnuts-warmup-inv-mass.csv" << std::endl;
  std::cout << "WROTE " << warmup_iterations << " FOR step_size TO "
	    << "build/walnuts-warmup-mass.csv" << std::endl;
  return 0;
}
