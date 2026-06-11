#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <stop_token>
#include <thread>
#include <vector>

#include "walnuts/concepts.hpp"
#include "walnuts/config.hpp"
#include "walnuts/spsc_buffer.hpp"
#include "walnuts/util.hpp"

namespace walnuts::detail {

/**
 * @brief A struct to represent a snapshot of the adaptation process
 * in a single chain.
 */
struct alignas(FALSE_SHARING_GUARD_SIZE) AdaptSnapshot {
  /**
   * @brief Construct an adaptation snapshot of size 0.
   */
  AdaptSnapshot() : AdaptSnapshot(0) {}

  /**
   * @brief Construct an adaptation snapshot of the given dimensionality.
   *
   * @param[in] dim The number of dimensions in the positions.
   */
  explicit AdaptSnapshot(Eigen::Index dim)
      : log_mass(Eigen::VectorXd::Constant(
            dim, std::numeric_limits<double>::quiet_NaN())),
        mass(Eigen::VectorXd::Constant(
            dim, std::numeric_limits<double>::quiet_NaN())) {}

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
 * @see `walnuts::SpscBuffer`
 * @see `walnuts::AdaptSnapshot`
 */
using Buffer = SpscBuffer<AdaptSnapshot>;

/**
 * @brief Return a deque of buffers of the given sizes.
 *
 * @param[in] num_chains The number of Markov chains.
 * @param[in] dim The number of dimensions.
 * @return The buffer container.
 */
inline std::deque<Buffer> construct_buffers(std::size_t num_chains,
                                            std::size_t dim) {
  std::deque<Buffer> buffers;
  auto snapshot = AdaptSnapshot(static_cast<Eigen::Index>(dim));
  for (std::size_t m = 0; m < num_chains; ++m) {
    buffers.emplace_back(snapshot);
  }
  return buffers;
}

/**
 * @brief A class that encapsulates the work done in a Markov chain for
 * embedding in a thread.
 *
 * @tparam AdaptiveSampler The base sampler being adapted.
 */
template <AdaptiveSampler A>
class AdaptWorker {
 public:
  /**
   * @brief Construct an adaptation worker with the specified configuration.
   *
   * @param[in] chain_id The identifier for which chain this is.
   * @param[in] init_cfg The initialization configuragation.
   * @param[in] warmup_cfg The configuration for warmup.
   * @param[in] buffer The buffer to hold the adaptation states.
   * @param[in] start_gate A latch to gate work to start synchronously across
   * workers.
   * @param[in] adapter The base sampler, methods of which are later called.
   */
  AdaptWorker(std::size_t chain_id, const InitConfig& init_cfg,
              const WarmupConfig& warmup_cfg, Buffer& buffer,
              std::latch& start_gate, A& adapter)
      : chain_id_(chain_id),
        init_config_(init_cfg),
        warmup_config_(warmup_cfg),
        buffer_(buffer),
        start_gate_(start_gate),
        adapter_(adapter) {}

  /**
   * @brief The functor called by the thread to do the adaptation.
   *
   * A call to this function first waits for the latch, then iterates
   * for a number of iterations between the specified min and max.
   * Adaptation terminates when the maximum number of iterations is
   * hit or a stop is requested through the stop token.  The thread
   * will yield based on the yield period specified in the warmup
   * configuration.
   *
   * @param[in] st The stop token for stopping the worker thread.
   */
  void operator()(const std::stop_token st) {
    interactive_qos();  // Apple silicon top priority; o.w. no-op
    start_gate_.get().arrive_and_wait();
    publish_snapshot(0);
    // from 1 so modulo ops don't trigger on first iteration
    std::size_t iter = 1;
    for (; iter <= warmup_config_.get().max_iter(); ++iter) {
      if (st.stop_requested()) {
        break;
      }
      if (iter % warmup_config_.get().yield_period() == 0) {
        std::this_thread::yield();
      }
      adapter_.get()();  // do the sampling
      if (iter % warmup_config_.get().publish_stride() == 0) {
        publish_snapshot(iter);
      }
    }
    if (iter % warmup_config_.get().publish_stride() != 0) {
      publish_snapshot(iter);
    }
  }

 private:
  void publish_snapshot(std::size_t iter) {
    AdaptSnapshot& snap = buffer_.get().write_buffer();
    snap.iter = iter;
    snap.log_step = adapter_.get().log_step_size();
    const auto lm = adapter_.get().log_mass();
    snap.log_mass = lm;
    snap.mass = lm.array().exp().matrix();
    buffer_.get().publish();
  }

  std::size_t chain_id_;
  std::reference_wrapper<const InitConfig> init_config_;
  std::reference_wrapper<const WarmupConfig> warmup_config_;
  std::reference_wrapper<Buffer> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
  std::reference_wrapper<A> adapter_;
};

/**
 * @brief A struct to hold the diagonal of the mass matrix and a step size.
 */
struct AdaptResult {
  /**
   * The average mass matrix.
   */
  Eigen::VectorXd mass_bar;

  /**
   * The average step size.
   */
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
template <InterruptCallback IC>
inline AdaptResult controller_loop(std::deque<Buffer>& buffers,
                                   std::vector<AdaptSnapshot>& latest,
                                   const IC& interrupt_callback,
                                   const InitConfig& init_cfg,
                                   const WarmupConfig& warmup_cfg) {
  std::size_t M = init_cfg.num_chains();
  std::size_t D = init_cfg.dims();

  Eigen::VectorXd mean_log_mass(D);
  Eigen::VectorXd geom_mean_mass(D);
  Eigen::VectorXd scratch_mass(D);
  while (true) {
    mean_log_mass.setZero();
    double mean_log_step = 0.0;
    std::size_t min_iter = std::numeric_limits<std::size_t>::max();
    for (std::size_t m = 0; m < M; ++m) {
      latest[m] = buffers[m].read_latest();
      min_iter = std::min(min_iter, latest[m].iter);
      mean_log_step += latest[m].log_step;  // means after division
      mean_log_mass += latest[m].log_mass;
    }
    mean_log_step /= static_cast<double>(M);
    mean_log_mass /= static_cast<double>(M);
    geom_mean_mass = mean_log_mass.array().exp().matrix();

    double max_rel_diff_mass = 0.0;
    double max_rel_diff_step = 0.0;
    double geom_mean_step = std::exp(mean_log_step);
    for (std::size_t m = 0; m < M; ++m) {
      double rel_diff_mass = l2_rel_diff(latest[m].mass, geom_mean_mass);
      max_rel_diff_mass = std::fmax(max_rel_diff_mass, rel_diff_mass);
      double chain_m_step = std::exp(latest[m].log_step);
      double rel_diff_step = (chain_m_step - geom_mean_step) / geom_mean_step;
      max_rel_diff_step = std::fmax(max_rel_diff_step, rel_diff_step);
    }

    bool enough_iters = min_iter >= warmup_cfg.min_iter();
    bool converged = enough_iters &&
                     max_rel_diff_mass <= warmup_cfg.mass_converge_tol() &&
                     max_rel_diff_step <= warmup_cfg.step_size_converge_tol();
    bool hit_max_iter = min_iter == warmup_cfg.max_iter();
    if (converged || hit_max_iter) {
      return {geom_mean_mass, std::exp(mean_log_step)};
    }

    interrupt_callback.throw_if_interrupted();
  }
}

/**
 * @brief The top-level function call for adaptation for the given configuration
 * and samplers.
 *
 * @tparam Adapter The type of adaptive sampler.
 * @tparam IC The type of the interrupt callback.
 * @param[in] init_cfg The initial configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[in,out] adapters The adaptive samplers for each chain.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 */
template <AdaptiveSampler A, InterruptCallback IC>
inline void adapt(const InitConfig& init_cfg, const WarmupConfig& warmup_cfg,
                  std::vector<A>& adapters, const IC& interrupt_callback) {
  std::deque<Buffer> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());
  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(AdaptWorker<A>(m, init_cfg, warmup_cfg, buffers[m],
                                        start_gate, adapters[m]));
  }
  std::vector<AdaptSnapshot> latest(init_cfg.num_chains());
  start_gate.arrive_and_wait();
  AdaptResult result;
  try {
    result = controller_loop(buffers, latest, interrupt_callback, init_cfg,
                             warmup_cfg);
  } catch (const std::exception& e) {
    for (auto& t : threads) {
      t.request_stop();
    }
    throw(e);
  }
  for (auto& t : threads) {
    t.request_stop();
  }
}

}  // namespace walnuts::detail
