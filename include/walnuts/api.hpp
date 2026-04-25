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
#include <walnuts/sampler.hpp>
#include <walnuts/triple_buffer.hpp>

namespace walnuts {

  template <typename Handler, typename LogProbGrad>
  std::vector<ChainRecord> walnuts(uint32_t seed, 
				   std::vector<Handler>& handlers,
				   const LogProbGrad& log_p_grad,
				   const InitConfig& init_cfg,
				   const WarmupConfig& warmup_cfg,
				   const SamplingConfig& sampling_cfg) {
    using AdaptiveSampler = nuts::AdaptiveWalnuts<LogProbGrad, double, std::mt19937, Handler>;
    using Sampler = nuts::WalnutsSampler<LogProbGrad, double, std::mt19937, Handler>;

    std::vector<AdaptiveSampler> adapters;
    adapters.reserve(init_cfg.num_chains());

    std::vector<std::mt19937> rngs(0);
    for (std::uint32_t m = 0; m < init_cfg.num_chains(); ++m) {
      std::seed_seq ss{seed, m + 1u};
      rngs.emplace_back(ss);
    }

    // TODO: unify to use external API config throughout
    for (std::uint32_t m = 0; m < init_cfg.num_chains(); ++m) {
      nuts::MassAdaptConfig<double>
	mass_cfg(init_cfg.mass(static_cast<size_t>(m)),
		 warmup_cfg.mass_init_count(),
		 warmup_cfg.mass_init_count(), // reuse init for iter offset
		 warmup_cfg.mass_additive_smoothing());
      
      nuts::AdamConfig<double>
	step_cfg(init_cfg.step_size(m),
		 warmup_cfg.step_accept_rate_target(),
		 warmup_cfg.step_learning_rate(),
		 warmup_cfg.step_gradient_decay(),
		 warmup_cfg.step_sq_gradient_decay(),
		 warmup_cfg.step_stabilization());

      nuts::WalnutsConfig<double>
	walnuts_cfg(sampling_cfg.max_hamiltonian_error(),
		    sampling_cfg.max_trajectory_doublings(),
		    sampling_cfg.max_step_halvings(),
		    sampling_cfg.min_micro_steps());
      
      std::seed_seq seed_m{seed, m + 1u};
      adapters
      	.emplace_back(AdaptiveSampler(rngs[m],
				      handlers[m],
				      log_p_grad,
				      init_cfg.position(static_cast<size_t>(m)),
				      mass_cfg,
				      step_cfg,
				      walnuts_cfg,
				      std::log2(warmup_cfg.max_macro_steps_target())));
					    
    }

    AdaptResult adapt_result
      = adapt<AdaptiveSampler>(init_cfg, warmup_cfg, adapters);


    // ********************DEBUG I/O*****************************
    std::cout << "\nSHARED ADAPTED RESULT:  "
	      << "stop_iter_min=" << adapt_result.stop_iter_min
	      << "  step_bar=" << adapt_result.step_bar
	      << "  ||mass_bar||=" << adapt_result.mass_bar.norm()
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
    // *************************************************


    std::vector<Sampler> samplers;
    for (std::size_t n = 0; n < adapters.size(); ++n)
      samplers.emplace_back(adapters[n].sampler());

    size_t num_rhat_evals = 0u;
    std::vector<ChainRecord> chain_records = sample(samplers,
						    sampling_cfg.rhat_converge_tol(),
						    sampling_cfg.max_iter(),
						    num_rhat_evals);

    
    // *********************** DEBUG I/O *******************
    std::cout << "\nnum Rhat evals = " << num_rhat_evals << "\n";
    std::size_t rows = 0;
    for (std::size_t m = 0; m < chain_records.size(); ++m) {
      const auto& chain_record = chain_records[m];
      std::size_t N_m = chain_record.num_draws();
      Eigen::VectorXd lps(N_m);
      for (std::size_t n = 0; n < N_m; ++n) {
	lps(static_cast<int64_t>(n)) = chain_record.logp(n);
      }
      rows += N_m;
      std::cout << "Chain " << m << "  count " << N_m
		<< "  mean(logp) " << lps.mean()
		<< "  sd(logp) [sample] " << std::sqrt(variance(lps)) << '\n';
    }
    std::cout << "Number of draws: " << rows << '\n';
    // *****************************************************

    
    return chain_records;
  }
  
}  // namespace walnuts
