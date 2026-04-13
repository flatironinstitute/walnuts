#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/validate.hpp>
#include <walnuts/util.hpp>

namespace nuts {

  
  struct InitConfig {
    template <typename RNG, typename LPG>
    InitConfig(RNG& rng,
    	       const LPG& lp_grad,
    	       uint64_t num_chains,
    	       uint64_t size,
    	       double step_size,
    	       double mass_smoothing,
    	       double init_scale)
      : step_sizes_(num_chains, step_size),
	positions_(num_chains),
	masses_(num_chains)
    {
      Random<double, RNG> rand(rng);
      for (size_t c = 0; c < num_chains; ++c) {
    	positions_[c] = init_scale * rand.standard_normal(size);
    	Eigen::VectorXd grad;
    	double lp;
    	lp_grad(positions_[c], lp, grad);
    	masses_[c] = (1 - mass_smoothing) * grad.array().abs().sqrt() + mass_smoothing;
        // masses_ = precisions_
	// variances_[c] = = precisions_[c].array().inverse().matrix();
      }
    }
    
    InitConfig(const std::vector<double>& step_sizes,
    	       const std::vector<Eigen::VectorXd>& positions,
    	       const std::vector<Eigen::VectorXd>& masses)
      : step_sizes_(step_sizes),
	positions_(positions),
	masses_(masses)
    {
      validate_positive(step_sizes, "InitConfig", "step_sizes");
      validate_positive(masses, "InitConfig", "masses");
      validate_same_size(step_sizes, positions, "InitConfig", "step_sizes", "positions");
      validate_same_size(step_sizes, masses, "InitConfig", "step_sizes", "masses");
    }

    std::vector<double> step_sizes_;
    std::vector<Eigen::VectorXd> positions_;
    std::vector<Eigen::VectorXd> masses_;
  };

  std::ostream& operator<<(std::ostream& out, const InitConfig& cfg) {
    out << "InitConfigs (by chain)"                                    << "\n";
    for (size_t n = 0; n < cfg.step_sizes_.size(); ++n) {
      if (n > 0) out << "\n";
      out << "  chain                    = " << n                      << "\n"
	  << "    cfg.step_sizes_.size() = " << cfg.step_sizes_.size() << "\n"
	  << "    step size              = " << cfg.step_sizes_[n]     << "\n"
	  << "    position               = " << cfg.positions_[n]      << "\n"
	  << "    mass                   = " << cfg.masses_[n]         << "\n";
    }
    return out;
  }
  

  struct WarmupConfig {
    WarmupConfig() { }

    WarmupConfig(uint64_t max_warmup_iter,
		 double step_size_converge_tol,
		 double mass_matrix_converge_tol,
		 double mass_init_count,
		 double mass_additive_smoothing,
		 double max_macro_steps_target,
		 double step_accept_rate_target,
		 double step_learning_rate,
		 double step_gradient_decay,
		 double step_sq_gradient_decay,
		 double step_stabilization)
      : max_warmup_iter_(max_warmup_iter),
	step_size_converge_tol_(step_size_converge_tol),
	mass_matrix_converge_tol_(mass_matrix_converge_tol),
	mass_init_count_(mass_init_count),
	mass_additive_smoothing_(mass_additive_smoothing),
	max_macro_steps_target_(max_macro_steps_target),
	step_accept_rate_target_(step_accept_rate_target),
	step_learning_rate_(step_learning_rate),
	step_gradient_decay_(step_gradient_decay),
	step_stabilization_(step_stabilization)
    { }
    
    uint64_t max_warmup_iter_ = 1000;

    double step_size_converge_tol_ = 0.1;
    double mass_matrix_converge_tol_ = 1.0;

    double mass_init_count_ = 4.0; 
    double mass_additive_smoothing_ = 1e-5;

    double max_macro_steps_target_ = 15.0;

    double step_accept_rate_target_ = 0.8;
    
    double step_learning_rate_ = 0.2;         // lambda
    double step_gradient_decay_ = 0.3;        // beta1
    double step_sq_gradient_decay_ = 0.99;    // beta2
    double step_stabilization_ = 1e-4;        // epsilon
  };

  std::ostream& operator<<(std::ostream& out, const WarmupConfig& cfg) {
    out << "WarmupConfig\n"
        << "  max_warmup_iter            = " << cfg.max_warmup_iter_            << "\n"
        << "  step_size_converge_tol     = " << cfg.step_size_converge_tol_     << "\n"
        << "  mass_matrix_converge_tol   = " << cfg.mass_matrix_converge_tol_   << "\n"
        << "  mass_init_count            = " << cfg.mass_init_count_            << "\n"
        << "  mass_additive_smoothing    = " << cfg.mass_additive_smoothing_    << "\n"
        << "  max_macro_steps_target     = " << cfg.max_macro_steps_target_     << "\n"
        << "  step_accept_rate_target    = " << cfg.step_accept_rate_target_    << "\n"
        << "  step_learning_rate         = " << cfg.step_learning_rate_         << "\n"
        << "  step_gradient_decay        = " << cfg.step_gradient_decay_        << "\n"
        << "  step_sq_gradient_decay     = " << cfg.step_sq_gradient_decay_     << "\n"
        << "  step_stabilization         = " << cfg.step_stabilization_         << "\n";
    return out;
  }
  
  struct SamplingConfig {
    SamplingConfig() { }

    SamplingConfig(uint64_t max_iter,
		   uint64_t max_trajectory_doublings,
		   uint64_t max_step_halvings,
		   double max_hamiltonian_error,
		   double rhat_converge_tol)
      : max_iter_(max_iter),
	max_trajectory_doublings_(max_trajectory_doublings),
	max_step_halvings_(max_step_halvings),
	max_hamiltonian_error_(max_hamiltonian_error),
	rhat_converge_tol_(rhat_converge_tol)
    { }

    uint64_t max_iter_ = 1000;
    uint64_t max_trajectory_doublings_ = 5;
    uint64_t max_step_halvings_ = 5;
    double max_hamiltonian_error_ = 0.5;
    double rhat_converge_tol_ = 1.01;
  };

  std::ostream& operator<<(std::ostream& out, const SamplingConfig& cfg) {
    out << "SamplingConfig\n"
        << "  max_iter                  = " << cfg.max_iter_                  << "\n"
        << "  max_trajectory_doublings  = " << cfg.max_trajectory_doublings_  << "\n"
        << "  max_step_halvings         = " << cfg.max_step_halvings_         << "\n"
        << "  max_hamiltonian_error     = " << cfg.max_hamiltonian_error_     << "\n"
        << "  rhat_converge_tol         = " << cfg.rhat_converge_tol_         << "\n";
    return out;
  }
  
  template <typename Handler, typename RNG, typename LogProbGrad>
  void walnuts(RNG& rng,
	       std::vector<Handler>& handlers,  
	       const LogProbGrad& log_p_grad,
	       const InitConfig& init_config,
	       const WarmupConfig& warmup_config,
	       const SamplingConfig& sampling_config) {
    std::cout << init_config << std::endl;
    std::cout << warmup_config << std::endl;
    std::cout << sampling_config << std::endl;
  }

}  // namespace walnuts
