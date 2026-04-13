#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "walnuts/api.hpp"

struct MyHandler {
  MyHandler(bool save_warmup = true)
    : save_warmup_(save_warmup) {
  }
      
  void on_warmup(const Eigen::VectorXd& position,
		 const double lp,
		 const double stepsize,
		 const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    warmup_draws_.push_back(position);
    warmup_lps_.push_back(lp);
    warmup_stepsizes_.push_back(stepsize);
    warmup_diag_inv_masses_.push_back(diag_inv_mass);
  }

  void on_warmup_complete(const double stepsize,
			  const Eigen::VectorXd& diag_inv_mass) {
    stepsize_ = stepsize;
    diag_inv_mass_ = diag_inv_mass;
  }

  void on_sample(const Eigen::VectorXd& position,
		 const double lp) {
    draws_.push_back(position);
    lps_.push_back(lp);
  }

  void on_stop() { }

  bool save_warmup_;
  double stepsize_ = 0;
  Eigen::VectorXd diag_inv_mass_ = Eigen::VectorXd();
  std::vector<Eigen::VectorXd> draws_ = std::vector<Eigen::VectorXd>();
  std::vector<double> lps_ = std::vector<double>();
  std::vector<Eigen::VectorXd> warmup_draws_ = std::vector<Eigen::VectorXd>();
  std::vector<double> warmup_lps_ = std::vector<double>();
  std::vector<double> warmup_stepsizes_ = std::vector<double>();
  std::vector<Eigen::VectorXd> warmup_diag_inv_masses_ = std::vector<Eigen::VectorXd>();
};

static void std_normal(const Eigen::VectorXd& x, double& lp,
		       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

static void demo_logp_grad() {
  std::cout << "DEMO: logp_grad(std_normal)" << std::endl;
  Eigen::VectorXd y(2);
  y << 1.2, -3.9;
  double lp;
  Eigen::VectorXd grad;
  std_normal(y, lp, grad);
  std::cout << "y = " << y
	    << "\n  lp=" << lp
	    << "\n  grad=" << grad
	    << std::endl;
}

template <typename RNG, typename LPG>
static nuts::InitConfig create_init(RNG& rng, const LPG& logp_grad, uint64_t chains, uint64_t dims) {
  double init_scale = 1.05;
  double mass_smoothing = 0.1;
  auto config = nuts::InitConfigBuilder(chains, dims)
    .positions(rng, init_scale)
    .masses(std_normal, mass_smoothing)
    .build();
  return config;
}

static nuts::WarmupConfig create_warmup() {
  auto config = nuts::WarmupConfigBuilder()
    .max_warmup_iter(200)
    .mass_init_count(2.5)
    .build();
  return config;
}

nuts::SamplingConfig create_sampling() {
  auto config = nuts::SamplingConfigBuilder()
    .max_iter(200)
    .max_trajectory_doublings(8)
    .build();
  return config;
}

int main() {
  uint32_t rng_seed = 42;
  std::mt19937 rng(rng_seed);
  uint64_t chains = 4;
  uint64_t dims = 2;

  
  // demo_logp_grad();

  auto logp_grad = std_normal;
  auto init_config = create_init(rng, logp_grad, chains, dims);
  auto warmup_config = create_warmup();
  auto sampling_config = create_sampling();
  std::cout << init_config     << "\n"
	    << warmup_config   << "\n"
	    << sampling_config << "\n";
  
  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}

