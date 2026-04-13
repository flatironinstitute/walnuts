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

template <typename RNG>
static nuts::InitConfig create_init(RNG& rng) {
  auto logp_grad = std_normal;
  uint64_t num_chains = 4;
  uint64_t size = 2;
  double step_size = 0.5;
  double inv_mass_smoothing = 0.25;
  double init_scale = 1.0;
  return {rng, logp_grad, num_chains, size, step_size,
      inv_mass_smoothing, init_scale};
}

static nuts::WarmupConfig create_warmup() {
  uint64_t max_warmup_iter = 1000;
  
  double step_size_converge_tol = 0.1;
  double mass_matrix_converge_tol = 1.0;

  double mass_init_count = 4.0; 
  double mass_additive_smoothing = 1e-5;

  double max_macro_steps_target = 15.0;

  double step_accept_rate_target = 0.8;
    
  double step_learning_rate = 0.2;        
  double step_gradient_decay = 0.3;       
  double step_sq_gradient_decay = 0.99;   
  double step_stabilization = 1e-4;       

  return {max_warmup_iter, step_size_converge_tol, mass_matrix_converge_tol,
      mass_init_count, mass_additive_smoothing, max_macro_steps_target,
      step_accept_rate_target, step_learning_rate, step_gradient_decay,
      step_sq_gradient_decay, step_stabilization};
}

nuts::SamplingConfig create_sampling() {
   uint64_t max_iter = 1000;
   uint64_t max_trajectory_doublings = 5;
   uint64_t max_step_halvings = 5;
   double max_hamiltonian_error = 0.5;
   double rhat_converge_tol = 1.01;
   return {max_iter, max_trajectory_doublings, max_step_halvings,
       max_hamiltonian_error, rhat_converge_tol};
}

int main() {
  uint32_t rng_seed = 42;
  std::mt19937 rng(rng_seed);
 
  // demo_logp_grad();

  auto init_config = create_init(rng);
  auto warmup_config = create_warmup();
  auto sampling_config = create_sampling();
  std::cout << init_config     << "\n"
	    << warmup_config   << "\n"
	    << sampling_config << "\n";
  
  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}

