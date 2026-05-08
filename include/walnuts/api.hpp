#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <walnuts/adapt.hpp>
#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/concepts.hpp>
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
 * @param[in] global_handler The handler for global cross-chain events.
 * @param[in] log_p_grad The log density and gradient function, called back.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[in] sampling_cfg The sampling configuration.
 * @throws std::invalid_argument If the number of handlers doesn't match
 * the initialization configuration's number of chains.
 */
template <ChainHandler H, GlobalHandler GH, InterruptCallback IC, LogpGrad F>
void walnuts(std::uint32_t seed, std::vector<H>& chain_handlers,
	     GH& global_handler, const IC& interrupt_callback,
	     const F& log_p_grad,
	     const InitConfig& init_cfg, const WarmupConfig& warmup_cfg,
	     const SamplingConfig& sampling_cfg) {
  using AdaptiveSampler = AdaptiveWalnuts<F, std::mt19937, H>;
  using Sampler = WalnutsSampler<F, std::mt19937, H>;

  if (chain_handlers.size() != init_cfg.num_chains()) {
    throw std::invalid_argument(
        "chain_handlers.size() must be equal to init_cfg.num_chains()");
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
    adapters.emplace_back(rngs[m], chain_handlers[m], log_p_grad,
                          init_cfg.position(m), init_cfg.init_chain_config(m),
                          warmup_cfg, sampling_cfg,
                          std::log2(warmup_cfg.max_macro_steps_target()));
  }
  adapt<AdaptiveSampler>(init_cfg, warmup_cfg, adapters, interrupt_callback);

  std::vector<Sampler> samplers;
  for (std::size_t n = 0; n < adapters.size(); ++n) {
    samplers.emplace_back(std::move(adapters[n].sampler()));
  }

  sample(samplers, global_handler, interrupt_callback,
	 sampling_cfg.rhat_converge_tol(),
         sampling_cfg.min_iter(), sampling_cfg.max_iter());
}

}  // namespace walnuts
