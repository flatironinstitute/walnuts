#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <walnuts/adapt.hpp>
#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/config.hpp>
#include <walnuts/sampler.hpp>
#include <walnuts/walnuts.hpp>

namespace walnuts {

/**
 * Return the chain records from running Walnuts with the specified
 * seed, sampling event handlers, and configuration.
 *
 * @tparam Handler The type of the event handlers.
 * @param[in] seed The seed for the pseudo-random number generator.
 * @param[in] handlers The collection of chain-specific handlers, which are
 * called back.
 * @param[in] log_p_grad The log density and gradient function, called back.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[in] sampling_cfg The sampling configuration.
 * @throws std::invalid_argument If the number of handlers doesn't match
 * the initialization configuration's number of chains.
 */
template <typename Handler, typename LogProbGrad>
void walnuts(uint32_t seed, std::vector<Handler>& handlers,
             const LogProbGrad& log_p_grad, const InitConfig& init_cfg,
             const WarmupConfig& warmup_cfg,
             const SamplingConfig& sampling_cfg) {
  using AdaptiveSampler =
      AdaptiveWalnuts<LogProbGrad, std::mt19937, Handler>;
  using Sampler = WalnutsSampler<LogProbGrad, double, std::mt19937, Handler>;

  if (handlers.size() != init_cfg.num_chains()) {
    throw std::invalid_argument(
        "handlers.size() must equal init_cfg.num_chains()");
  }

  std::vector<std::mt19937> rngs(0);
  rngs.reserve(init_cfg.num_chains());
  for (std::uint32_t m = 0; m < init_cfg.num_chains(); ++m) {
    std::seed_seq ss{seed, m + 1u};
    rngs.emplace_back(ss);
  }

  std::vector<AdaptiveSampler> adapters;
  adapters.reserve(init_cfg.num_chains());
  for (std::uint64_t m = 0; m < init_cfg.num_chains(); ++m) {
    adapters.emplace_back(AdaptiveSampler(
        rngs[m], handlers[m], log_p_grad,
        init_cfg.position(static_cast<size_t>(m)),
        init_cfg.init_chain_config(static_cast<std::size_t>(m)), warmup_cfg,
        sampling_cfg, std::log2(warmup_cfg.max_macro_steps_target())));
  }

  AdaptResult adapt_result =
      adapt<AdaptiveSampler>(init_cfg, warmup_cfg, adapters);

  // ********************ADAPTATION DEBUG I/O*****************************
  std::cout << "\nSHARED ADAPTED RESULT:  "
            << "  step_bar=" << adapt_result.step_bar
            << "  ||mass_bar||=" << adapt_result.mass_bar.norm() << '\n';
  std::cout << "\nPER CHAIN FINAL STATES:\n";
  for (std::size_t m = 0; m < adapters.size(); ++m) {
    std::cout << m << ")"
              << " iter = " << adapters[m].iter()
              << "  step = " << std::exp(adapters[m].log_step_size())
              << "  ||log_mass|| = " << adapters[m].log_mass().norm()
              << std::endl;
  }
  // *************************************************

  std::vector<Sampler> samplers;
  for (std::size_t n = 0; n < adapters.size(); ++n) {
    samplers.emplace_back(std::move(adapters[n].sampler()));
  }

  std::size_t num_rhat_evals{0};
  double rhat;
  sample(samplers, sampling_cfg.rhat_converge_tol(),
	 sampling_cfg.min_iter(), sampling_cfg.max_iter(),
         num_rhat_evals, rhat);

  // *********************** SAMPLING DEBUG I/O *******************
  std::cout << "\nnum Rhat evals = " << num_rhat_evals << ";  Rhat = " << rhat
            << "\n";
  // *****************************************************
}

}  // namespace walnuts
