#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <latch>
#include <limits>
#include <numeric>
#include <stop_token>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <walnuts/concepts.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/util.hpp>

namespace walnuts {

/**
 * @brief Return the sum of the sizes in the vector.
 *
 * @param[in] xs The vector to sum.
 * @return The sum.
 */
std::size_t sum(const std::vector<std::size_t>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), std::size_t(0),
                               std::plus<>{}, std::identity());
}

/**
 * @brief Return the bias-adjusted sample variance estimate.
 *
 * @param[in] xs The vector whose variance is required.
 * @return The variance.
 */
double variance(const Eigen::VectorXd& xs) noexcept {
  return (xs.array() - xs.mean()).square().mean()
    / static_cast<double>((xs.size() - 1));
}

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
  float sample_mean;

  /** The within-chain variance. */
  float sample_var;

  /** The chain length. */
  std::uint32_t count;
};

/**
 * @brief An atomically-wrapped `ChainStats` object with helper methods.
 *
 * Usable as a single-producer, single-consumer (SPSC) store.
 */
class AtomicChainStats {
 public:
  /**
   * Construct an atomically-wrapped `ChainStats` object initialized
   * to `NaN` for the mean and variance and zero for the count and store
   * it in `relaxed` order.
   */
  AtomicChainStats() noexcept {
    ChainStats init{std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(), 0u};
    data_.store(init, std::memory_order_relaxed);
  }

  /**
   * @brief Store the specified object with the specified memory order.
   *
   * The default uses a conservate `release` order.
   *
   * @param[in] p The chain statistics object to store.
   * @param[in] mem_order The memory order for the storage.
   */
  void store(const ChainStats& p,
             std::memory_order mem_order = std::memory_order_release) noexcept {
    data_.store(p, mem_order);
  }

  /**
   * @brief Return a copy of the local chain stats with the specified memory
   * order.
   *
   * The default uses a conservate `acquire` order.
   *
   * @param[in] mem_order The memory order for the storage.
   * @return Copy of the local chain statistics object.
   */
  ChainStats load(
      std::memory_order mem_order = std::memory_order_acquire) const noexcept {
    return data_.load(mem_order);
  }

 private:
  std::atomic<ChainStats> data_;
};

/**
 * @brief Accumulator for online mean and smaple variance calculations.
 *
 * Welford's algorithm stores sufficient statistics with which to
 * compute a running mean and sample variance The accumulator stores
 * only three sufficient statistics: a `std::size_t` and two `double`
 * values.  The algorithm is more numerically stable for variance
 * calculations than the naive algorithm.
 */
class WelfordAccumulator {
 public:
  /**
   * @brief Construct an accumulator with no observed values.
   */
  WelfordAccumulator() : n_(0), mean_(0.0), M2_(0.0) {}

  /**
   * @brief Observe a value.
   *
   * @param[in] x The observed value.
   */
  void observe(double x) {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / static_cast<double>(n_);
    const double delta2 = x - mean_;
    M2_ += delta * delta2;
  }

  /**
   * @brief Return the number of values observed.
   *
   * @return The number of values observed.
   */
  std::size_t count() const { return n_; }

  /**
   * @brief Return the mean of all of the values observed, or 0
   * if no values have been observed.
   *
   * @return The mean of the observed values.
   */
  double mean() const { return mean_; }

  /**
   * @brief Return the sample variance of the observed values.
   *
   * The sample variance is the unbiased estimator of variance.
   * It divides by number of observations minus one.  Thus if
   * there have been fewer than two observations, the sample
   * variance is undefined and `NaN` will be returned.
   *
   * @return The sample variance of the observed values.
   */
  double sample_variance() const {
    return n_ > 1 ? (M2_ / static_cast<double>(n_ - 1))
                  : std::numeric_limits<double>::quiet_NaN();
  }

  /**
   * @brief Return the sample statistics object for a chain.
   *
   * The returned object has reduced precision: `float` instead of
   * `double` and `uint32_t` instead of `std::size_t`).
   *
   * @return The sample statistics object for a chain.
   */
  ChainStats sample_stats() {
    return {static_cast<float>(mean()), static_cast<float>(sample_variance()),
            static_cast<uint32_t>(count())};
  }

  /**
   * @brief Reset the accumulator to its initial state of having seen
   * zero observations.
   */
  void reset() {
    n_ = 0;
    mean_ = 0.0;
    M2_ = 0.0;
  }

 private:
  std::size_t n_;
  double mean_;
  double M2_;
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
   * @param[out] acs The atomic chain statistics for this chain.
   * @param[in,out] start_gate The latch gating work and monitoring.
   * @param[in] yield_period The period in iterations of this thread
   * yielding.
   */
  ChainWorker(std::size_t min_draws, std::size_t max_draws, S& sampler,
              AtomicChainStats& acs, std::latch& start_gate,
              std::size_t yield_period = 1024)
      : min_draws_(min_draws),
        max_draws_(max_draws),
        sampler_(sampler),
        acs_(acs),
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
    for (std::size_t iter = 1;
         iter <= max_draws_ && !(iter >= min_draws_ && st.stop_requested());
         ++iter) {
      if (iter % yield_period_ == 0) {
        std::this_thread::yield();
      }
      double lp = sampler_.get()();
      logp_stats_.observe(lp);
      acs_.get().store(logp_stats_.sample_stats());
    }
  }

 private:
  const std::size_t min_draws_;
  const std::size_t max_draws_;
  std::reference_wrapper<S> sampler_;
  WelfordAccumulator logp_stats_;
  std::reference_wrapper<AtomicChainStats> acs_;
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
template <GlobalHandler GH>
static void controller_loop(
    std::vector<Padded<AtomicChainStats>>& stats_by_chain,
    GH& global_handler, double rhat_threshold,
    std::latch& start_gate, std::size_t min_draws_per_chain,
    std::size_t max_draws_per_chain,
    std::chrono::milliseconds eval_period = std::chrono::milliseconds{10}) {
  initiated_qos();  // Apple silicon second-highest priority; o.w. no-op
  const std::size_t M = stats_by_chain.size();
  Eigen::VectorXd chain_means(M);
  Eigen::VectorXd chain_variances(M);
  std::vector<std::size_t> counts(M, 0);

  start_gate.wait();
  auto next = std::chrono::steady_clock::now() + eval_period;
  while (true) {
    bool achieved_min_draws = true;
    for (std::size_t m = 0; m < M && achieved_min_draws; ++m) {
      ChainStats u = stats_by_chain[m].val.load();
      counts[m] = u.count;
      if (counts[m] < min_draws_per_chain) {
        achieved_min_draws = false;
      }
      chain_means(static_cast<Eigen::Index>(m)) =
          static_cast<double>(u.sample_mean);
      chain_variances(static_cast<Eigen::Index>(m)) =
          static_cast<double>(u.sample_var);
    }
    if (achieved_min_draws) {
      double variance_of_means = variance(chain_means);
      double mean_of_variances = chain_variances.mean();
      double r_hat = std::sqrt(1 + variance_of_means / mean_of_variances);
      global_handler.on_r_hat(r_hat);
      std::size_t num_draws = sum(counts);
      if (r_hat <= rhat_threshold || num_draws == M * max_draws_per_chain) {
        break;
      }
    }
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
 * @tparam Sampler The type of the sampler.
 * @param[in] samplers The vector of samplers.
 * @param[in,out] global_handler The global event handler for sampling.
 * @param[in] rhat_threshold The threshold below which sampling is stopped.
 * @param[in] min_draws_per_chain The minimum number of draws per chain.
 * @param[in] max_draws_per_chain The maximum number of draws per chain.
 */
template <Sampler S, GlobalHandler GH>
void sample(std::vector<S>& samplers, GH& global_handler,
            double rhat_threshold, std::size_t min_draws_per_chain,
            std::size_t max_draws_per_chain) {
  std::size_t M = samplers.size();
  std::vector<Padded<AtomicChainStats>> stats_by_chain(M);
  std::latch start_gate(static_cast<std::ptrdiff_t>(M));
  std::vector<std::jthread> threads;
  threads.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    threads.emplace_back(
        ChainWorker<S>(min_draws_per_chain, max_draws_per_chain,
		       samplers[m], stats_by_chain[m].val, start_gate));
  }
  controller_loop(stats_by_chain, global_handler, rhat_threshold, start_gate,
                  min_draws_per_chain, max_draws_per_chain);
}

}  // namespace walnuts
