// examples/adapt_monitor.cpp

// clang++ -std=c++20 -march=native -O3 -pthread -I ../include adapt_monitor.cpp -o adapt_monitor
// ./adapt_monitor

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

#include "walnuts/padded.hpp"
#include "walnuts/triple_buffer.hpp"


struct AdaptSnapshot {
  std::uint32_t iter = 0;
  float log_step = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> log_mass;
  std::vector<float> mass;

  explicit AdaptSnapshot(std::size_t dim)
      : log_mass(dim, std::numeric_limits<float>::quiet_NaN()),
        mass(dim, std::numeric_limits<float>::quiet_NaN()) {}
};

using Buffer = walnuts::TripleBuffer<AdaptSnapshot>;
using PaddedBuffer = walnuts::Padded<Buffer>;

static PaddedBuffer construct_buffer(std::size_t dim) {
  auto make = [dim] { return AdaptSnapshot(dim); };
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

struct AdaptConfig {
  std::size_t dim = 0;
  std::size_t num_chains = 0;
  std::size_t max_warmup_iters = 0;

  std::uint32_t min_iters = 20;
  std::uint32_t publish_stride = 5;
  std::chrono::milliseconds probe_period{10};

  float tau_mass = 1e-2f;
  float tau_step = 1e-2f;

  std::uint32_t yield_period = 64;
};

template <class AdaptiveSampler>
class AdaptWorker {
 public:
  AdaptWorker(std::uint32_t chain_id,
              const AdaptConfig& cfg,
              PaddedBuffer& buffer,
              std::latch& start_gate,
              AdaptiveSampler adapter)
      : chain_id_(chain_id),
        cfg_(cfg),
        buffer_(buffer.val),
        start_gate_(start_gate),
        adapter_(std::move(adapter)) {}

  void operator()(std::stop_token st) {
    start_gate_.arrive_and_wait();
    std::uint32_t last_done = 0;
    publish_snapshot(0);
    for (std::uint32_t iter = 1; iter <= cfg_.max_warmup_iters; ++iter) {
      if (st.stop_requested()) break;
      if (cfg_.yield_period > 0 && (iter % cfg_.yield_period == 0)) {
        std::this_thread::yield();
      }
      adapter_.step(iter);
      last_done = iter;
      if (cfg_.publish_stride > 0 && (iter % cfg_.publish_stride == 0)) {
        publish_snapshot(iter);
      }
    }
    if (cfg_.publish_stride == 0 || (last_done % cfg_.publish_stride != 0)) {
      publish_snapshot(last_done);
    }
  }

 private:
  void publish_snapshot(std::uint32_t iter) {
    AdaptSnapshot& snap = buffer_.write_buffer();
    snap.iter = iter;
    snap.log_step = (iter == 0) ? std::numeric_limits<float>::quiet_NaN()
                                : adapter_.log_step();

    const auto lm = adapter_.log_mass();
    for (std::size_t d = 0; d < cfg_.dim; ++d) {
      const float v = (iter == 0) ? std::numeric_limits<float>::quiet_NaN() : lm[d];
      snap.log_mass[d] = v;
      snap.mass[d] = std::exp(v);
    }
    buffer_.publish();
  }

  std::uint32_t chain_id_;
  AdaptConfig cfg_;
  Buffer& buffer_;
  std::latch& start_gate_;
  AdaptiveSampler adapter_;
};

template <class AdaptiveSampler>
class AdaptRunner {
 public:
  AdaptRunner(std::uint32_t chain_id,
              const AdaptConfig& cfg,
              PaddedBuffer& buffer,
              std::latch& start_gate,
              AdaptiveSampler adapter)
      : worker_(chain_id, cfg, buffer, start_gate, std::move(adapter)),
        thread_(std::ref(worker_)) {}

  AdaptRunner(const AdaptRunner&) = delete;
  AdaptRunner& operator=(const AdaptRunner&) = delete;
  AdaptRunner(AdaptRunner&&) noexcept = delete;
  AdaptRunner& operator=(AdaptRunner&&) noexcept = delete;

  void request_stop() noexcept { thread_.request_stop(); }
  void join() { thread_.join(); }

 private:
  AdaptWorker<AdaptiveSampler> worker_;
  std::jthread thread_;
};


struct AdaptResult {
  std::vector<float> mass_bar;
  float step_bar = std::numeric_limits<float>::quiet_NaN();
  std::uint32_t stop_iter_min = 0;
};

static float l2_norm(std::span<const float> x) noexcept {
  double sum_sq = 0.0;
  for (float v : x) {
    const double dv = static_cast<double>(v);
    sum_sq += dv * dv;
  }
  return static_cast<float>(std::sqrt(sum_sq));
}

static void elementwise_exp(std::span<const float> log_x,
                            std::span<float> out_x) noexcept {
  const std::size_t n = log_x.size();
  for (std::size_t i = 0; i < n; ++i) {
    out_x[i] = std::exp(log_x[i]);
  }
}

static void elementwise_add(std::span<const float> x,
                            std::span<float> acc) noexcept {
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; ++i) {
    acc[i] += x[i];
  }
}

static void elementwise_scale(float a, std::span<float> x) noexcept {
  for (float& v : x) {
    v *= a;
  }
}

static float l2_diff(std::span<const float> a,
                     std::span<const float> b) noexcept {
  double sum_sq = 0.0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum_sq += diff * diff;
  }
  return static_cast<float>(std::sqrt(sum_sq));
}

template <class Stopper>
static AdaptResult controller_loop(std::vector<PaddedBuffer>& buffers,
                                   const AdaptConfig& cfg,
                                   Stopper stop_all) {
  const std::size_t M = cfg.num_chains;
  const std::size_t D = cfg.dim;

  std::vector<float> mean_log_mass(D, 0.0f);
  std::vector<float> mean_mass(D, 0.0f);
  std::vector<float> scratch_mass(D, 0.0f);

  float mean_log_step = 0.0f;

  std::uint32_t min_iter = 0;

  auto next = std::chrono::steady_clock::now() + cfg.probe_period;
  std::vector<const AdaptSnapshot*> latest(M, nullptr);
  while (true) {
    std::fill(mean_log_mass.begin(), mean_log_mass.end(), 0.0f);
    mean_log_step = 0.0f;
    min_iter = std::numeric_limits<std::uint32_t>::max();

    // Read latest snapshots (asynchronously).
    for (std::size_t m = 0; m < M; ++m) {
      latest[m] = &buffers[m].val.read_latest();
      const AdaptSnapshot& s = *latest[m];
      min_iter = std::min(min_iter, s.iter);
      mean_log_step += s.log_step;
      elementwise_add(std::span<const float>(s.log_mass), mean_log_mass);
    }

    mean_log_step /= static_cast<float>(M);
    elementwise_scale(1.0f / static_cast<float>(M), mean_log_mass);
    elementwise_exp(std::span<const float>(mean_log_mass),
                    std::span<float>(mean_mass));

    const float mean_mass_norm = l2_norm(std::span<const float>(mean_mass));

    float max_rel_mass = 0.0f;
    float max_rel_step = 0.0f;

    // Compute max relative errors.
    for (std::size_t m = 0; m < M; ++m) {
      const AdaptSnapshot& s =*latest[m]; //  buffers[m].val.read_latest();

      // Mass comparison on linear scale: ||mass_m - mean_mass|| / ||mean_mass||
      const float diff_mass =
          l2_diff(std::span<const float>(s.mass), std::span<const float>(mean_mass));
      const float rel_mass = diff_mass / mean_mass_norm;
      max_rel_mass = std::max(max_rel_mass, rel_mass);

      // Step comparison on log scale: |log_step_m - mean_log_step| / |mean_log_step|
      const float rel_step =
          std::abs(s.log_step - mean_log_step) / std::abs(mean_log_step);
      max_rel_step = std::max(max_rel_step, rel_step);
    }

    // Optional progress print (single line).
    std::cout << '\r'
              << "min_iter=" << min_iter
              << "  max_rel_mass=" << max_rel_mass
              << "  max_rel_step=" << max_rel_step
              << std::flush;

    const bool enough_iters = (min_iter >= cfg.min_iters);
    const bool converged =
        enough_iters && (max_rel_mass <= cfg.tau_mass) &&
        (max_rel_step <= cfg.tau_step);

    const bool hit_max = (min_iter >= cfg.max_warmup_iters);

    if (converged || hit_max) {
      stop_all();
      std::cout << '\n';
      AdaptResult out;
      out.mass_bar = std::move(mean_mass);
      out.step_bar = std::exp(mean_log_step);
      out.stop_iter_min = min_iter;
      return out;
    }

    std::this_thread::sleep_until(next);
    next += cfg.probe_period;
  }
}

// ********** EXAMPLE AFTER HERE **************

// this is a stub that will get replaced with WALNUTS
class ExampleSampler {
 public:
  ExampleSampler(std::size_t dim, std::uint64_t seed)
      : dim_(dim),
        rng_(seed),
        z_(0.0f, 1.0f),
        log_mass_means_(means(dim)),
        log_mass_(dim),
        log_step_(std::log(0.1f)) {}

  std::size_t dim() const noexcept { return dim_; }

  void step(std::uint32_t iter) {
    const float sd = 1.0f / std::sqrt(static_cast<float>(iter));
    log_step_ = std::log(0.1f) + sd * z_(rng_);
    for (std::size_t d = 0; d < dim_; ++d) {
      log_mass_[d] = log_mass_means_[d] + sd * z_(rng_);
    }
  }

  float log_step() const noexcept { return log_step_; }

  std::span<const float> log_mass() const noexcept { return log_mass_; }

 private:
  static std::vector<float> means(std::size_t dim) {
    std::vector<float> m(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      const float x = static_cast<float>(d + 1);
      m[d] = std::log(x * x);
    }
    return m;
  }

  std::size_t dim_;
  std::mt19937_64 rng_;
  std::normal_distribution<float> z_;
  std::vector<float> log_mass_means_;
  std::vector<float> log_mass_;
  float log_step_;
};


int main() {
  std::cout << "Adaptation Monitoring Demo" << std::endl;

  AdaptConfig cfg;
  cfg.dim = 100;
  cfg.num_chains = 32;
  cfg.max_warmup_iters = 2000;
  cfg.min_iters = 20;
  cfg.publish_stride = 5;
  cfg.probe_period = std::chrono::milliseconds{1};
  cfg.tau_mass = 1e1f;
  cfg.tau_step = 1e1f;

  std::vector<PaddedBuffer> buffers = construct_buffers(cfg.num_chains, cfg.dim);

  std::latch start_gate(static_cast<std::ptrdiff_t>(cfg.num_chains));
  std::stop_source stop_source;

  std::vector<std::unique_ptr<AdaptRunner<ExampleSampler>>> runners;
  runners.reserve(cfg.num_chains);

  std::mt19937_64 seeder(1234);
  for (std::size_t m = 0; m < cfg.num_chains; ++m) {
    runners.emplace_back(std::make_unique<AdaptRunner<ExampleSampler>>(
      static_cast<std::uint32_t>(m), cfg, buffers[m], start_gate,
      ExampleSampler(cfg.dim, seeder())));
  }
  auto stop_all = [&] {
    for (auto& r : runners) {
      r->request_stop();
    }
  };
  
  AdaptResult res = controller_loop(buffers, cfg, stop_all);

  for (auto& r : runners) {
    r->join();
  }

  std::cout << "stop_iter_min=" << res.stop_iter_min
            << "  step_bar=" << res.step_bar
            << "  ||mass_bar||=" << l2_norm(std::span<const float>(res.mass_bar))
            << '\n';

  const float mass_bar_norm = l2_norm(std::span<const float>(res.mass_bar));
  const float log_step_bar = std::log(res.step_bar);

  std::cout << "Per-chain final state:\n";
  for (std::size_t m = 0; m < cfg.num_chains; ++m) {
    const AdaptSnapshot& s = buffers[m].val.read_latest();
    const float step = std::exp(s.log_step);
    const float mass_norm = l2_norm(std::span<const float>(s.mass));
    const float rel_mass = l2_diff(std::span<const float>(s.mass),
				   std::span<const float>(res.mass_bar)) / mass_bar_norm;
    const float rel_step = std::abs(s.log_step - log_step_bar) / std::abs(log_step_bar);
    std::cout << "  chain " << m
	      << "  iter=" << s.iter
	      << "  step=" << step
	      << "  ||mass||=" << mass_norm
	      << "  mass_norm=" << mass_norm
	      << "  rel_mass=" << rel_mass
	      << '\n';
  }  
  return 0;
}
