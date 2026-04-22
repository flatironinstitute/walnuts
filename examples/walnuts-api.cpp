
#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "walnuts/api.hpp"
#include "walnuts/config.hpp"

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

int main() {
  auto logp_grad = std_normal;

  uint32_t seed = 48;
  std::seed_seq seed_seq_for_init{seed, 0u};
  std::mt19937 rng{seed_seq_for_init};
  uint64_t num_chains = 32;
  uint64_t dims = 10;
  
  std::vector<MyHandler> handlers(num_chains);
  for (size_t n = 0; n < num_chains; ++n) {
    handlers[n] = MyHandler();
  }

  auto init_cfg = walnuts::InitConfigBuilder(num_chains, dims)
    .positions(rng, 1.05)
    .masses(std_normal, 0.1)
    .build();

  auto warmup_cfg = walnuts::WarmupConfigBuilder()
    .min_max_iter(50, 2000)
    .step_size_converge_tol(1)
    .mass_init_count(4.0)
    .build();

  auto sampling_cfg = walnuts::SamplingConfigBuilder()
    .min_max_iter(10, 200)
    .max_trajectory_doublings(8)
    .build();

  // std::cout << init_cfg << std::endl;
  // std::cout << warmup_cfg << std::endl;
  // std::cout << sampling_cfg << std::endl;
  
  walnuts::walnuts(seed,
		   handlers,
		   logp_grad,
		   init_cfg, warmup_cfg, sampling_cfg);
  
  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}

