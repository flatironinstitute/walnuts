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
#include <stop_token>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/config.hpp>

#include <walnuts/config.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/triple_buffer.hpp>

namespace walnuts {

  struct alignas(walnuts::CACHE_LINE_SIZE) AdaptSnapshot {
    std::uint64_t iter = 0;
    double log_step = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd log_mass;
    Eigen::VectorXd mass;

    explicit AdaptSnapshot(std::int64_t dim)
      : log_mass(dim), mass(dim) {
      log_mass = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::quiet_NaN());
      mass = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::quiet_NaN());
    }
  };

  using Buffer = walnuts::TripleBuffer<AdaptSnapshot>;
  using PaddedBuffer = walnuts::Padded<Buffer>;

  static PaddedBuffer construct_buffer(std::size_t dim) {
    auto make = [dim] { return AdaptSnapshot(static_cast<std::int64_t>(dim)); };
    return PaddedBuffer{Buffer(make)};
  }

  static std::vector<PaddedBuffer> construct_buffers(std::size_t num_chains,
						     std::size_t dim) {
    std::vector<PaddedBuffer> buffers;
    buffers.reserve(num_chains);
    for (std::size_t m = 0; m < num_chains; ++m) {
      buffers.emplace_back(construct_buffer(dim));
    }
    return buffers;
  }

  template <class AdaptiveSampler>
  class AdaptWorker {
  public:
    AdaptWorker(std::uint64_t chain_id,
		const walnuts::InitConfig& init_cfg,
		const walnuts::WarmupConfig& warmup_cfg,
		PaddedBuffer& buffer,
		std::latch& start_gate,
		AdaptiveSampler& adapter)
      : chain_id_(chain_id),
        init_config_(init_cfg),
	warmup_config_(warmup_cfg),
        buffer_(buffer.val),
        start_gate_(start_gate),
        adapter_(adapter) {}

    void operator()(std::stop_token st) {
      start_gate_.get().arrive_and_wait();
      std::uint64_t last_done = 0;
      publish_snapshot(0);
      for (std::uint32_t iter = 1; iter <= warmup_config_.max_iter(); ++iter) {
	if (st.stop_requested()) break;
	if (warmup_config_.yield_period() > 0 && (iter % warmup_config_.yield_period() == 0)) {
	  std::this_thread::yield();
	}
	adapter_.get()();  // actually do the sampling!
	adapter_.get().step_size();
	last_done = iter;
	if (warmup_config_.publish_stride() > 0 && (iter % warmup_config_.publish_stride() == 0)) {
	  publish_snapshot(iter);
	}
      }
      if (warmup_config_.publish_stride() == 0 || (last_done % warmup_config_.publish_stride() != 0)) {
	publish_snapshot(last_done);
      }
    }

  private:
    void publish_snapshot(std::uint64_t iter) {
      AdaptSnapshot& snap = buffer_.get().write_buffer();
      snap.iter = iter;
      snap.log_step = (iter == 0) ? std::numeric_limits<double>::quiet_NaN()
	: adapter_.get().log_step_size();

      const auto lm = adapter_.get().log_mass();
      for (std::int64_t d = 0; d < static_cast<std::int64_t>(init_config_.dims()); ++d) {
	const double v = (iter == 0) ? std::numeric_limits<double>::quiet_NaN() : lm[d];
	snap.log_mass(d) = v;
	snap.mass(d) = std::exp(v);
      }
      buffer_.get().publish();
    }

    std::uint64_t chain_id_;
    const walnuts::InitConfig& init_config_;
    const walnuts::WarmupConfig& warmup_config_;
    std::reference_wrapper<Buffer> buffer_;
    std::reference_wrapper<std::latch> start_gate_;
    std::reference_wrapper<AdaptiveSampler> adapter_;
  };


  struct AdaptResult {
    AdaptResult(const Eigen::VectorXd& mass_b, double step_b, std::uint64_t stop_iter_m):
      mass_bar(mass_b),
      step_bar(step_b),
      stop_iter_min(stop_iter_m) {
    }
    Eigen::VectorXd mass_bar;
    double step_bar; 
    std::uint64_t stop_iter_min;
  };


  static double l2_rel_diff(const Eigen::VectorXd& a,
			    const Eigen::VectorXd& b) noexcept {
    return ((a - b).array() / b.array()).matrix().norm();
  }


  template <class Stopper>
  static AdaptResult controller_loop(std::vector<PaddedBuffer>& buffers,
				     const walnuts::InitConfig& init_cfg,
				     const walnuts::WarmupConfig& warmup_cfg,
				     Stopper stop_all) {
    const std::size_t M = init_cfg.num_chains();
    const std::size_t D = init_cfg.dims();

    Eigen::VectorXd mean_log_mass = Eigen::VectorXd::Zero(static_cast<int64_t>(D));
    Eigen::VectorXd mean_mass = Eigen::VectorXd::Zero(static_cast<int64_t>(D));
    Eigen::VectorXd scratch_mass = Eigen::VectorXd::Zero(static_cast<int64_t>(D));

    double mean_log_step = 0.0;

    std::uint64_t min_iter = 0;

    auto probe_period = std::chrono::microseconds(warmup_cfg.probe_microseconds());

    auto next = std::chrono::steady_clock::now() + probe_period;
    std::vector<const AdaptSnapshot*> latest(M, nullptr);
    while (true) {
      std::fill(mean_log_mass.begin(), mean_log_mass.end(), 0.0);
      mean_log_step = 0.0;
      min_iter = std::numeric_limits<std::uint32_t>::max();
      for (std::size_t m = 0; m < M; ++m) {
	latest[m] = &buffers[m].val.read_latest();
	const AdaptSnapshot& s = *latest[m];
	min_iter = std::min(min_iter, s.iter);
	mean_log_step += s.log_step;
	mean_log_mass += s.log_mass;
      }

      mean_log_step /= static_cast<double>(M);
      mean_log_mass /= static_cast<double>(M);
      mean_mass = mean_log_mass.array().exp().matrix();

      double max_rel_mass = 0.0;
      double max_rel_step = 0.0;

      for (std::size_t m = 0; m < M; ++m) {
	const AdaptSnapshot& s = *latest[m]; //  buffers[m].val.read_latest();

	const double diff_mass = l2_rel_diff(s.mass, mean_mass);
	max_rel_mass = std::fmax<double>(max_rel_mass, diff_mass);

	// Step comparison on log scale: |log_step_m - mean_log_step| / |mean_log_step|
	double s_step = static_cast<double>(std::exp(s.log_step));
	double m_step = static_cast<double>(std::exp(mean_log_step));
	double rel_step = (s_step - m_step) / m_step;
	max_rel_step = std::fmax<double>(max_rel_step, rel_step);
      }

      const bool enough_iters = (min_iter >= warmup_cfg.min_iter());
      const bool converged =
	enough_iters
	&& max_rel_mass <= warmup_cfg.mass_converge_tol()
	&& max_rel_step <= warmup_cfg.step_size_converge_tol();

      const bool hit_max = min_iter >= warmup_cfg.max_iter();

      if (converged || hit_max) {
	stop_all();
	return {mean_mass, std::exp(mean_log_step), min_iter};
      }

      std::this_thread::sleep_until(next);
      next += probe_period;
    }
  }
  
  template <typename Adapter>
  AdaptResult adapt(const walnuts::InitConfig& init_cfg,
		    const walnuts::WarmupConfig& warmup_cfg,
		    std::vector<Adapter>& adapters) {
    std::vector<PaddedBuffer> buffers = construct_buffers(init_cfg.num_chains(), init_cfg.dims());
    std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));

    std::vector<std::jthread> threads;
    threads.reserve(init_cfg.num_chains());
    for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
      std::uint32_t chain_id = static_cast<std::uint32_t>(m);
      threads.emplace_back(AdaptWorker<Adapter>(chain_id, init_cfg, warmup_cfg,
						buffers[m], start_gate, adapters[m]));
    }

    start_gate.arrive_and_wait();

    auto stop_all = [&] {
      for (auto& t : threads) {
	t.request_stop();
      }
      for (auto& t : threads) {
	t.join();
      }
    };
    return controller_loop(buffers, init_cfg, warmup_cfg, stop_all);
  }
  
} // namespace nuts
