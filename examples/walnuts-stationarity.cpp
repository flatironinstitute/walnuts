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
  std::size_t D = 20;
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(D));
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 0.1;
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

  unsigned int seed = 428763;
  std::mt19937 rng(seed);

  std::normal_distribution std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (std::size_t i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  std::cout << "\nADAPTIVE WALNUTS" << std::endl;
  nuts::AdaptiveWalnuts walnuts(rng, normal_logp_grad, theta_init, mass_cfg,
                                step_cfg, walnuts_cfg);

  std::size_t warmup_iterations = 1000;
  for (std::size_t n = 0; n < warmup_iterations; ++n) {
    walnuts();
  }

  auto file_name = "walnuts-stationarity.csv";
  std::ofstream out(file_name);
  out << std::fixed << std::setprecision(8);
  std::size_t sampling_iterations = 10000000;
  auto sampler = walnuts.sampler();
  for (std::size_t n = 0; n < sampling_iterations; ++n) {
    if (n % 1000000 == 0) {
      std::cout << "   iteration: " << n
		<< "/" << sampling_iterations << std::endl;
    }
    Eigen::VectorXd theta = sampler();
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
