#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <walnuts/adaptive_walnuts.hpp>

static void normal_logp_grad(const Eigen::VectorXd& x, double& logp,
			     Eigen::VectorXd& grad) {
  const auto D = x.size();
  grad = Eigen::VectorXd::Zero(D);
  logp = 0;
  for (auto d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
}

int main() {
  std::size_t D = 200;
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(D));
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 1e-5;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);

 double step_size_init = 0.5;
  double target_accept_rate = 0.8;  // min 2.0 / 3.0
  double learn_rate = 0.2;
  double beta1 = 0.3;
  double beta2 = 0.99;
  double epsilon = 1e-4;
  nuts::AdamConfig<double> step_cfg(step_size_init, target_accept_rate, learn_rate,
				    beta1, beta2, epsilon);

  double max_error = 1.0;  // 61% Metropolis
  std::size_t max_nuts_depth = 10;
  std::size_t max_step_depth = 8;
  std::size_t min_micro_steps = 1;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth, min_micro_steps);
  
  unsigned int seed = 8735487;
  std::mt19937 rng(seed);

  std::normal_distribution std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (std::size_t i = 0; i < D; ++i) {
    theta_init(static_cast<Eigen::Index>(i)) = std_normal(rng);
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

  std::size_t warmup_iterations = 10000;
  for (std::size_t n = 0; n < warmup_iterations; ++n) {
    walnuts();

    Eigen::VectorXd inv_mass_diag = walnuts.inv_mass();
    for (std::size_t d = 0; d < D; ++d) {
      if (d > 0) {
	out_inv_mass << ',';
      }
      out_inv_mass << inv_mass_diag(static_cast<Eigen::Index>(d));
    }
    out_inv_mass << '\n';

    double step_size = walnuts.step_size();
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
