#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/validate.hpp>

namespace walnuts {

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

    bool save_warmup_ = true;
    double stepsize_ = 0;
    Eigen::VectorXd diag_inv_mass_ = Eigen::VectorXd();
    std::vector<Eigen::VectorXd> draws_ = std::vector<Eigen::VectorXd>();
    std::vector<double> lps_ = std::vector<double>();
    std::vector<Eigen::VectorXd> warmup_draws_ = std::vector<Eigen::VectorXd>();
    std::vector<double> warmup_lps_ = std::vector<double>();
    std::vector<double> warmup_stepsizes_ = std::vector<double>();
    std::vector<Eigen::VectorXd> warmup_diag_inv_masses_ = std::vector<Eigen::VectorXd>();
  };

  struct SampleConfig {
    uint64_t max_sampling_iter = 1000;
    uint64_t max_nuts_trajectory_doublings = 5;
    uint64_t max_walnuts_step_halvings = 5;
    double max_walnuts_hamiltonian_error = 0.5;
    double rhat_converge_tol_ = 1.01;
  };

  struct WarmupConfig {
    uint64_t max_warmup_iter = 1000;

    double step_size_converge_tol_ = 0.1;
    double mass_matrix_converge_tol_ = 1.0;

    double mass_init_count_ = 4.0; 
    double mass_additive_smoothing_ = 1e-5;

    double max_macro_steps_target_ = 15.0;

    double step_accept_rate_target_ = 0.8;
    
    double adam_step_learning_rate_ = 0.2;       // lambda
    double adam_step_gradient_decay_ = 0.3;      // beta1
    double adam_step_sq_gradient_decay_ = 0.99;  // beta2
    double adam_step_stabilization_ = 1e-4;        // epsilon
  };

  struct InitConfig {
    InitConfig(const std::vector<double>& step_sizes,
    	       const std::vector<Eigen::VectorXd>& positions,
    	       const std::vector<Eigen::VectorXd>& inv_masses)
      : init_step_sizes_(step_sizes),
	init_positions_(positions),
	init_inv_masses_(inv_masses)
    {
      validate_same_size(step_sizes, positions, "InitConfig", "step_sizes", "positions");
      validate_same_size(step_sizes, inv_masses, "InitConfig", "step_sizes", "inv_masses");
      validate_positive(step_sizes, "InitConfig", "step_sizes");
      validate_positive(inv_masses, "InitConfig", "inv_masses");
    }
    

    template <typename RNG>
    InitConfig(RNG& rng,
	       const uint64_t chains,
	       const uint64_t dim,
	       const double stepsize=0.5)
      : InitConfig(std::vector<double>(chains, stepsize),
		   std::vector<Eigen::VectorXd>(chains,
						Eigen::VectorXd::Zero(dim)),
		   std::vector<Eigen::VectorXd>(chains,
						Eigen::VectorXd::Ones(dim))) {
    }

    // InitConfig(double init_step_size,
    // 	       std::vector<Eigen::VectorXd>& positions) { }
    
    // template <typename LPG>
    // InitConfig(uint64_t dim,
    // 	       double random_init_scale,
    // 	       const LPG& log_prob_grad) { }

    const std::vector<double> init_step_sizes_;
    const std::vector<Eigen::VectorXd> init_positions_;
    const std::vector<Eigen::VectorXd> init_inv_masses_;
  };
  
  template <typename Handler, typename RNG, typename LogProbGrad>
  void walnuts(RNG& rng,
	       std::vector<Handler>& handlers,  
	       const LogProbGrad& log_p_grad,
	       const std::vector<InitConfig>& init_configs,  // implies number chains & dims
	       const WarmupConfig& warmup_config,
	       const SampleConfig& sample_config) {
  }

  
}  // namespace walnuts
