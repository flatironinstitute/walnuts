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
template <typename Handler, typename GlobalHandler, typename LogProbGrad>
void walnuts(std::uint32_t seed, std::vector<Handler>& handlers,
	     GlobalHandler& global_handler,
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
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    adapters.emplace_back(
        rngs[m], handlers[m], log_p_grad,
        init_cfg.position(m),
        init_cfg.init_chain_config(m), warmup_cfg,
        sampling_cfg, std::log2(warmup_cfg.max_macro_steps_target()));
  }
  AdaptResult adapt_result =
      adapt<AdaptiveSampler>(init_cfg, warmup_cfg, adapters);

  std::vector<Sampler> samplers;
  for (std::size_t n = 0; n < adapters.size(); ++n) {
    samplers.emplace_back(std::move(adapters[n].sampler()));
  }

  sample(samplers, global_handler, sampling_cfg.rhat_converge_tol(),
	 sampling_cfg.min_iter(), sampling_cfg.max_iter());
}

}  // namespace walnuts
