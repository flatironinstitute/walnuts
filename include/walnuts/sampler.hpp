#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <latch>
#include <stop_token>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "walnuts/concepts.hpp"
#include "walnuts/online_moments.hpp"
#include "walnuts/spsc_buffer.hpp"
#include "walnuts/util.hpp"

namespace walnuts::detail {

/**
 * @brief A struct to hold the within chain summary statistics for
 * R-hat.
 *
 * Members are defined to be `float` and `uint32_t` in order to make
 * the whole lock-free on common architectures when wrapped with
 * `std::atomic`.
 */
struct ChainStats {
  /** The within-chain mean. */
  double sample_mean;

  /** The within-chain variance. */
  double sample_var;

  /** The chain length. */
  std::size_t count;
};

/**
 * @brief The worker functor for threads that calls the sampler and updates
 * the accumulator.
 *
 * @tparam Sampler The sampling functor.
 */
template <Sampler S>
class ChainWorker {
 public:
  /**
   * @brief Construct a woker to embed in a thread for sampling.
   *
   * @param[in] min_draws The minimum number of draws.
   * @param[in] max_draws The maximum number of draws.
   * @param[in] sampler The sampler.
   * @param[out] buffer The buffer of chain statistics for this chain.
   * @param[in,out] start_gate The latch gating work and monitoring.
   * @param[in] yield_period The period in iterations of this thread
   * yielding.
   */
  ChainWorker(std::size_t min_draws, std::size_t max_draws, S& sampler,
              SpscBuffer<ChainStats>& buffer, std::latch& start_gate,
              std::size_t yield_period = 1024)
      : min_draws_(min_draws),
        max_draws_(max_draws),
        sampler_(sampler),
        buffer_(buffer),
        start_gate_(start_gate),
        yield_period_(yield_period) {}

  /**
   * @brief Called by `std::jthread` to do the sampling.
   *
   * The operator returns if a stop is requested, if the maximum
   * number of draws have been sampled.  It yields according to
   * the period of the chain worker.  It calls the running
   * statistics acucmulator used for monitoring.
   *
   * @param[in] st The `jthread` stop token to track if a stop has been
   * requested externally.
   */
  void operator()(const std::stop_token st) {
    interactive_qos();  // Apple silicon top priority; o.w. no-op
    start_gate_.get().arrive_and_wait();
    for (std::size_t iter = 1; iter <= max_draws_ && !st.stop_requested();
         ++iter) {
      if (iter % yield_period_ == 0) {
        std::this_thread::yield();
      }
      double lp = sampler_.get()();
      logp_stats_.observe(lp);
      ChainStats chain_stats{logp_stats_.mean(), logp_stats_.sample_variance(),
                             logp_stats_.count()};
      buffer_.get().write_buffer() = chain_stats;
      buffer_.get().publish();
    }
  }

 private:
  const std::size_t min_draws_;
  const std::size_t max_draws_;
  std::reference_wrapper<S> sampler_;
  WelfordAccumulator logp_stats_;
  std::reference_wrapper<SpscBuffer<ChainStats>> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
  const std::size_t yield_period_;
};

/**
 * @brief The function called by the controller to monitor the threads
 * for the chains.
 *
 * @param[in,out] stats_by_chain The per-chain objects holding the
 * statistics to monitor.
 * @param[in] rhat_threshold When R-hat falls below this value, sampling
 * terminates.
 * @param[in,out] start_gate The latch gating the monitor and thread workers
 * to synchronize starting.
 * @param[in] min_draws_per_chain The minimum number of iterations per chain.
 * @param[in] max_draws_per_chain The maximum number of iterations per chain.
 * @param[in] eval_period The period between initiating cross-chain R-hat
 * calculations.
 */
template <GlobalHandler GH, InterruptCallback IC>
static void controller_loop(
    std::vector<SpscBuffer<ChainStats>>& stats_by_chain, GH& global_handler,
    const IC& interrupt_callback, double rhat_threshold, std::latch& start_gate,
    std::size_t min_draws_per_chain, std::size_t max_draws_per_chain,
    std::chrono::nanoseconds eval_period = std::chrono::milliseconds{1}) {
  interactive_qos();  // Apple silicon highest priority; o.w. no-op
  const std::size_t M = stats_by_chain.size();
  Eigen::VectorXd chain_means(M);
  Eigen::VectorXd chain_variances(M);
  std::vector<std::size_t> counts(M, 0);

  start_gate.arrive_and_wait();
  auto next = std::chrono::steady_clock::now() + eval_period;
  while (true) {
    bool achieved_min_draws = true;
    for (std::size_t m = 0; m < M && achieved_min_draws; ++m) {
      const ChainStats& u = stats_by_chain[m].read_latest();
      counts[m] = u.count;
      if (counts[m] < min_draws_per_chain) {
        achieved_min_draws = false;
      }
      chain_means(static_cast<Eigen::Index>(m)) = u.sample_mean;
      chain_variances(static_cast<Eigen::Index>(m)) = u.sample_var;
    }
    if (achieved_min_draws) {
      double variance_of_means = variance(chain_means);
      double mean_of_variances = chain_variances.mean();
      double r_hat = std::sqrt(1 + variance_of_means / mean_of_variances);
      global_handler.on_r_hat(r_hat);
      std::size_t num_draws = sum(counts);
      if (r_hat <= rhat_threshold || num_draws == M * max_draws_per_chain) {
        return;
      }
    }
    interrupt_callback.throw_if_interrupted();
    std::this_thread::sleep_until(next);
    next += eval_period;
  }
}

/**
 * @brief Sample from the specified samplers in parallel until the
 * R-hat threshold is attained.
 *
 * The `Sampler` object must implement `double operator()()` to return
 * log density of latest sample and call chain local handlers
 * indirectly through the samplers, and also implement `size_t dim()`.
 *
 * @tparam S The type of the sampler.
 * @tparam GH The type of the global handler.
 * @tparam IC The type of the interrupt callback.
 * @param[in] samplers The vector of samplers.
 * @param[in,out] global_handler The global event handler for sampling.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 * @param[in] rhat_threshold The threshold below which sampling is stopped.
 * @param[in] min_draws_per_chain The minimum number of draws per chain.
 * @param[in] max_draws_per_chain The maximum number of draws per chain.
 */
template <Sampler S, GlobalHandler GH, InterruptCallback IC>
inline void sample(std::vector<S>& samplers, GH& global_handler,
                   const IC& interrupt_callback, double rhat_threshold,
                   std::size_t min_draws_per_chain,
                   std::size_t max_draws_per_chain) {
  std::size_t M = samplers.size();
  std::vector<SpscBuffer<ChainStats>> stats_by_chain(M);
  std::latch start_gate(static_cast<std::ptrdiff_t>(M + 1));
  std::vector<std::jthread> threads;
  threads.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    threads.emplace_back(ChainWorker<S>(min_draws_per_chain,
                                        max_draws_per_chain, samplers[m],
                                        stats_by_chain[m], start_gate));
  }
  try {
    controller_loop(stats_by_chain, global_handler, interrupt_callback,
                    rhat_threshold, start_gate, min_draws_per_chain,
                    max_draws_per_chain, std::chrono::milliseconds{1});
  } catch (const std::exception& e) {
    for (auto& t : threads) {
      t.request_stop();
    }
    throw(e);
  }
}

}  // namespace walnuts::detail
