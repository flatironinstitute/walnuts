#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <random>
#include <stop_token>
#include <thread>
#include <vector>
#include <walnuts/config.hpp>

#include <walnuts/config.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/triple_buffer.hpp>
#include <walnuts/util.hpp>

namespace walnuts {

/**
 * @brief A struct to represent a snapshot of the adaptation process
 * in a single chain.
 */
struct alignas(CACHE_LINE_SIZE) AdaptSnapshot {

  /**
   * @brief Construct an adaptation snapshot of size 0.
   */
  AdaptSnapshot() : AdaptSnapshot(0) { }
  
  /**
   * @brief Construct an adaptation snapshot of the given dimensionality.
   *
   * @param[in] dim The number of dimensions in the positions.
   */
  explicit AdaptSnapshot(Eigen::Index dim) :
    log_mass(Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::quiet_NaN())),
    mass(Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::quiet_NaN()))
    {   }


  /** The number of iterations carried out in the chain. */
  std::size_t iter = 0;

  /** The currently adapted log step size. */
  double log_step = std::numeric_limits<double>::quiet_NaN();

  /** The currently adapted log mass matirx. */
  Eigen::VectorXd log_mass;

  /** The currently adapted mass matrix. */
  Eigen::VectorXd mass;
};

/**
 * A triple buffer of adaptation snapshots.
 *
 * @see TripleBuffer
 * @see AdaptSnapshot
 */
using Buffer = TripleBuffer<AdaptSnapshot>;

/**
 * A padded triple buffer of adaptation snapshots.
 *
 * @see Padded
 * @see Buffer
 */
using PaddedBuffer = Padded<Buffer>;

/**
 * @brief Return a padded buffer with the specified dimensionality.
 *
 * @param[in] dim The dimensionality of the positions.
 * @return A padded buffer of the specified dimensionality.
 */
static PaddedBuffer construct_buffer(std::size_t dim) {
  auto make = [dim] { return AdaptSnapshot(static_cast<Eigen::Index>(dim)); };
  return PaddedBuffer{Buffer(make)};
}

/**
 * @brief Return a vector of padded buffers of the given sizes.
 *
 * @param[in] num_chains The number of Markov chains.
 * @param[in] dim The number of dimensions.
 * @return The padded buffer container.
 */
static std::vector<PaddedBuffer> construct_buffers(std::size_t num_chains,
                                                   std::size_t dim) {
  std::vector<PaddedBuffer> buffers;
  buffers.reserve(num_chains);
  for (std::size_t m = 0; m < num_chains; ++m) {
    buffers.emplace_back(construct_buffer(dim));
  }
  return buffers;
}

/**
 * @brief A class that encapsulates the work done in a Markov chain for
 * embedding in a thread.
 *
 * @tparam AdaptiveSampler The base sampler being adapted.
 */
template <class AdaptiveSampler>
class AdaptWorker {
 public:
  /**
   * @brief Construct an adaptation worker with the specified configuration.
   *
   * @param[in] chain_id The identifier for which chain this is.
   * @param[in] init_cfg The initialization configuragation.
   * @param[in] warmup_cfg The configuration for warmup.
   * @param[in] buffer The padded buffer to hold the adaptation states.
   * @param[in] start_gate A latch to gate work to start synchronously across
   * workers.
   * @param[in] adapter The base sampler, methods of which are later called.
   */
  AdaptWorker(std::size_t chain_id, const InitConfig& init_cfg,
              const WarmupConfig& warmup_cfg, PaddedBuffer& buffer,
              std::latch& start_gate, AdaptiveSampler& adapter)
      : chain_id_(chain_id),
        init_config_(init_cfg),
        warmup_config_(warmup_cfg),
        buffer_(buffer.val),
        start_gate_(start_gate),
        adapter_(adapter) {}

  /**
   * @brief The functor doing the work.
   *
   * A call to this function first waits for the latch, then iterates
   * up the warmup config specified max iterations.  If a stop is
   * requested through the stop token, it stops working.  It yields
   * according to the yield period in the warmup configuration.  Then
   * it actually does the sampling, gets the step size, gets the mass
   * matrix, and publishes the adapted state if it matches the stride
   * specified by the warmup configuration.  When it's done looping, it
   * publishes a final snapshot of the chain's warmup state.
   *
   * @param[in] st The stop token for stopping the worker thread.
   */
  void operator()(const std::stop_token st) {
    start_gate_.get().arrive_and_wait();
    publish_snapshot(0);
    std::size_t iter = 1;  // from 1 so modulo ops don't rstart at 1 so % ops don't trigger on first iteration
    for (; iter <= warmup_config_.max_iter(); ++iter) {
      if (iter >= warmup_config_.min_iter() && st.stop_requested()) {
        break;  // should we be waiting for min_iter when stop requested?
      }
      if (iter % warmup_config_.yield_period() == 0) {
        std::this_thread::yield();
      }
      adapter_.get()();  // do the sampling
      if (iter % warmup_config_.publish_stride() == 0) {
        publish_snapshot(iter);
      }
    }
    if (iter % warmup_config_.publish_stride() != 0) {
      publish_snapshot(iter);
    }
  }

 private:
  void publish_snapshot(std::size_t iter) {
    AdaptSnapshot& snap = buffer_.get().write_buffer();
    snap.iter = iter;
    snap.log_step = adapter_.get().log_step_size();
    const auto lm = adapter_.get().log_mass();
    for (Eigen::Index d = 0; d < static_cast<Eigen::Index>(init_config_.dims()); ++d) {
      snap.log_mass(d) = lm[d];
      snap.mass(d) = std::exp(lm[d]);
    }
    buffer_.get().publish();
  }

  std::size_t chain_id_;
  const InitConfig& init_config_;
  const WarmupConfig& warmup_config_;
  std::reference_wrapper<Buffer> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
  std::reference_wrapper<AdaptiveSampler> adapter_;
};

/**
 * @brief A struct to hold matrix and step size for a chain.
 */
struct AdaptResult {
  Eigen::VectorXd mass_bar;
  double step_bar;
};

/**
 * @brief The implementation of the control monitor with the adaptation
 * state of each chain and configuration.
 *
 * @param[in,out] buffers The adaptation state of all the chains.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @return Statistics for the completed adaptation process.
 */
static AdaptResult controller_loop(std::vector<PaddedBuffer>& buffers,
				   std::vector<AdaptSnapshot>& latest,
                                   const InitConfig& init_cfg,
                                   const WarmupConfig& warmup_cfg) {
  const std::size_t M = init_cfg.num_chains();
  const std::size_t D = init_cfg.dims();

  auto probe_period =
      std::chrono::microseconds(warmup_cfg.probe_microseconds());
  auto next = std::chrono::steady_clock::now() + probe_period;

  Eigen::VectorXd mean_log_mass(D);
  Eigen::VectorXd geom_mean_mass(D);
  Eigen::VectorXd scratch_mass(D);
  while (true) {
    mean_log_mass.setZero();
    double mean_log_step = 0.0;
    std::size_t min_iter = std::numeric_limits<std::size_t>::max();
    for (std::size_t m = 0; m < M; ++m) {
      latest[m] = buffers[m].val.read_latest();
      min_iter = std::min(min_iter, latest[m].iter);
      mean_log_step += latest[m].log_step;  // means after division
      mean_log_mass += latest[m].log_mass;
    }
    mean_log_step /= static_cast<double>(M);
    mean_log_mass /= static_cast<double>(M);
    geom_mean_mass = mean_log_mass.array().exp().matrix();

    double max_rel_diff_mass = 0.0;
    double max_rel_diff_step = 0.0;
    for (std::size_t m = 0; m < M; ++m) {
      double rel_diff_mass = l2_rel_diff(latest[m].mass, geom_mean_mass);
      max_rel_diff_mass = std::fmax(max_rel_diff_mass, rel_diff_mass);
      double chain_m_step = static_cast<double>(std::exp(latest[m].log_step));
      double geom_mean_step = std::exp(mean_log_step);
      double rel_diff_step = (chain_m_step - geom_mean_step) / geom_mean_step;
      max_rel_diff_step = std::fmax(max_rel_diff_step, rel_diff_step);
    }

    const bool enough_iters = min_iter >= warmup_cfg.min_iter();
    const bool converged = enough_iters &&
                           max_rel_diff_mass <= warmup_cfg.mass_converge_tol() &&
                           max_rel_diff_step <= warmup_cfg.step_size_converge_tol();
    const bool hit_max_iter = min_iter >= warmup_cfg.max_iter();
    if (converged || hit_max_iter) {
      return {geom_mean_mass, std::exp(mean_log_step)};
    }

    std::this_thread::sleep_until(next);
    next += probe_period;
  }
}

/**
 * @brief The top-level function call for adaptation for the given configuration
 * and samplers.
 *
 * @tparam Adapter The type of adaptive sampler.
 * @param[in] init_cfg The initial configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[in,out] adapters The adaptive samplers for each chain.
 * @return The completed adaptation configuration.
 */
template <typename Adapter>
AdaptResult adapt(const InitConfig& init_cfg, const WarmupConfig& warmup_cfg,
                  std::vector<Adapter>& adapters) {
  std::vector<PaddedBuffer> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());
  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(AdaptWorker<Adapter>(
        m, init_cfg, warmup_cfg, buffers[m], start_gate, adapters[m]));
  }
  std::vector<AdaptSnapshot> latest(init_cfg.num_chains());
  start_gate.arrive_and_wait();
  AdaptResult result = controller_loop(buffers, latest, init_cfg, warmup_cfg);
  for (auto& t : threads) {
    t.request_stop();
  }
  for (auto& t : threads) {
    t.join();
  }
  return result;
}

}
