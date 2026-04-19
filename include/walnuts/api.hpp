#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/config.hpp>

#include <walnuts/adapt.hpp>
#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/config.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/triple_buffer.hpp>

namespace walnuts {

  // AdaptiveWalnuts
  // VectorXd operator()()  // sample
  // WalnutsSampler sampler() // return sampler
  // VectorXd inv_mass()
  // double step_size()
  // size_t min_micro_steps()

  // size_t dim() const noexcept
  // double step_size() const noexcept
  // double log_step_size() const noexcept
  // VectorXd log_mass() const noexcept
  // size_t iter() const noexcept
  // --------------------
  
  // TemporaryStubAdapter(rng, dim)
  // size_t dim() const noexcept;
  // void step(size_t iter);
  // double log_step() const noexcept;
  // VectorXd log_mass() const noexcept;
  // size_t iter() const noexcept;

  template <typename RNG>
  class TemporaryStubAdapter {
  public:
    TemporaryStubAdapter(RNG&& rng, std::size_t dim)
      : rng_(rng),
	dim_(dim),
	iter_(0),
        z_(0.0, 1.0),
        log_mass_means_(means(dim)),
        log_mass_(dim),
        log_step_(std::log(0.1)) {}

    std::size_t dim() const noexcept { return dim_; }

    void step_size() {
      ++iter_;  // here because no opertor()() called for sampling
      double sd = 1 / std::sqrt(static_cast<double>(iter_));
      log_step_ = std::log(0.1) + sd * z_(rng_);
      for (std::int64_t d = 0; d < static_cast<std::int64_t>(dim_); ++d) {
    	log_mass_(d) = log_mass_means_(d) + sd * z_(rng_);
      }
    }

    double log_step_size() const noexcept { return log_step_; }

    Eigen::VectorXd log_mass() const noexcept { return log_mass_; }

    std::size_t iter() const noexcept { return iter_; }
  
  private:
    static Eigen::VectorXd means(std::size_t dim) {
      Eigen::VectorXd m(dim);
      for (std::int64_t d = 0; d < static_cast<std::int64_t>(dim); ++d) {
	const double x = static_cast<double>(d + 1);
	m(d) = std::log(x * x);
      }
      return m;
    }

    RNG rng_;  // intentionally keeping key
    const std::size_t dim_;
    std::size_t iter_;
    std::normal_distribution<double> z_;
    Eigen::VectorXd log_mass_means_;
    Eigen::VectorXd log_mass_;
    double log_step_;
  };
  

  template <typename Handler, typename LogProbGrad>
  void walnuts(uint32_t seed, 
	       std::vector<Handler>& , // handlers,  
	       const LogProbGrad& log_p_grad,
	       const InitConfig& init_cfg,
	       const WarmupConfig& warmup_cfg,
	       const SamplingConfig& sampling_cfg) {

    std::cout << init_cfg << std::endl;
    std::cout << warmup_cfg << std::endl;
    std::cout << sampling_cfg << std::endl;


    std::vector<nuts::AdaptiveWalnuts<LogProbGrad, double, std::mt19937>> walnuts_adapters;
    walnuts_adapters.reserve(init_cfg.num_chains());
    
    for (std::uint32_t m = 0; m < init_cfg.num_chains(); ++m) {
      std::seed_seq seed_m{seed, m + 1u};
      std::mt19937 rng{seed_m};
      nuts::MassAdaptConfig<double> mass_cfg(init_cfg.mass(static_cast<size_t>(m)),
					     warmup_cfg.mass_init_count(),
					     warmup_cfg.mass_init_count(), // reuse init for iter offset
					     warmup_cfg.mass_additive_smoothing());
      
      nuts::AdamConfig<double> step_cfg(init_cfg.step_size(m),
					warmup_cfg.step_accept_rate_target(),
					warmup_cfg.step_learning_rate(),
					warmup_cfg.step_gradient_decay(),
					warmup_cfg.step_sq_gradient_decay(),
					warmup_cfg.step_stabilization());

      std::size_t min_micro_steps= 1;
      nuts::WalnutsConfig<double> walnuts_cfg(sampling_cfg.max_hamiltonian_error(),
					      sampling_cfg.max_trajectory_doublings(),
					      sampling_cfg.max_step_halvings(),
					      min_micro_steps);
      
      walnuts_adapters
      	.emplace_back(nuts::AdaptiveWalnuts(rng,
      					    log_p_grad,
      					    init_cfg.position(static_cast<size_t>(m)),
      					    mass_cfg,
      					    step_cfg,
      					    walnuts_cfg,
      					    std::log2(warmup_cfg.max_macro_steps_target())));
					    
    }
      
    std::vector<TemporaryStubAdapter<std::mt19937>> adapters;
    adapters.reserve(init_cfg.num_chains());
    auto D = init_cfg.dims();
    for (std::uint32_t m = 0; m < init_cfg.num_chains(); ++m) {
      std::seed_seq seed_m{seed, m + 1u};
      std::mt19937 rng{seed_m};
      adapters.emplace_back(TemporaryStubAdapter(std::move(rng), D));
    }

    AdaptResult res
      = adapt<TemporaryStubAdapter<std::mt19937>>(init_cfg, warmup_cfg, adapters);

    const double mass_bar_norm = res.mass_bar.norm();
  
    std::cout << "\nSHARED ADAPTED RESULT:  "
	      << "stop_iter_min=" << res.stop_iter_min
	      << "  step_bar=" << res.step_bar
	      << "  ||mass_bar||=" << mass_bar_norm
	      << '\n';

    std::cout << "\nPER CHAIN FINAL STATES:\n";
    for (std::size_t m = 0; m < adapters.size(); ++m) {
      std::cout << m << ")"
		<< " iter = " << adapters[m].iter()
		<< "  step = " << std::exp(adapters[m].log_step_size())
		<< "  ||log_mass|| = "
		<< adapters[m].log_mass().norm()
		<< std::endl;

    }
  }
  
}  // namespace walnuts
