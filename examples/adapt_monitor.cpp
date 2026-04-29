// examples/adapt_monitor.cpp

// clang++ -std=c++20 -march=native -O3 -pthread -I ../include -I
// ../build/_deps/eigen adapt_monitor.cpp -o adapt_monitor
// ./adapt_monitor

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
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include <walnuts/config.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/triple_buffer.hpp>

struct alignas(walnuts::DI_SIZE) AdaptSnapshot {
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

template <class AdaptiveSampler>
class AdaptWorker {
 public:
  AdaptWorker(std::uint32_t chain_id, const nuts::InitConfig& init_cfg,
              const nuts::WarmupConfig& warmup_cfg, PaddedBuffer& buffer,
              std::latch& start_gate, AdaptiveSampler& adapter)
      : chain_id_(chain_id),
        init_config_(init_cfg),
        warmup_config_(warmup_cfg),
        buffer_(buffer.val),
        start_gate_(start_gate),
        adapter_(adapter) {}

  void operator()(std::stop_token st) {
    start_gate_.get().arrive_and_wait();
    std::uint32_t last_done = 0;
    publish_snapshot(0);
    for (std::uint32_t iter = 1; iter <= warmup_config_.max_iter(); ++iter) {
      if (st.stop_requested()) {
        break;
      }
      if (warmup_config_.yield_period() > 0 &&
          (iter % warmup_config_.yield_period() == 0)) {
        std::this_thread::yield();
      }
      adapter_.get().step(iter);
      last_done = iter;
      if (warmup_config_.publish_stride() > 0 &&
          (iter % warmup_config_.publish_stride() == 0)) {
        publish_snapshot(iter);
      }
    }
    if (warmup_config_.publish_stride() == 0 ||
        (last_done % warmup_config_.publish_stride() != 0)) {
      publish_snapshot(last_done);
    }
  }

 private:
  void publish_snapshot(std::uint32_t iter) {
    AdaptSnapshot& snap = buffer_.get().write_buffer();
    snap.iter = iter;
    snap.log_step = (iter == 0) ? std::numeric_limits<float>::quiet_NaN()
                                : adapter_.get().log_step();

    const auto lm = adapter_.get().log_mass();
    for (std::size_t d = 0; d < init_config_.dims(); ++d) {
      const float v =
          (iter == 0) ? std::numeric_limits<float>::quiet_NaN() : lm[d];
      snap.log_mass[d] = v;
      snap.mass[d] = std::exp(v);
    }
    buffer_.get().publish();
  }

  std::uint32_t chain_id_;
  const nuts::InitConfig& init_config_;
  const nuts::WarmupConfig& warmup_config_;
  std::reference_wrapper<Buffer> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
  std::reference_wrapper<AdaptiveSampler> adapter_;
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

static float l2_rel_diff(std::span<const float> a,
                         std::span<const float> b) noexcept {
  double sum_sq = 0.0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    double ad = static_cast<double>(a[i]);
    double bd = static_cast<double>(b[i]);
    double rel_diff = (ad - bd) / bd;
    sum_sq += rel_diff * rel_diff;
  }
  return static_cast<float>(std::sqrt(sum_sq));
}

template <class Stopper>
static AdaptResult controller_loop(std::vector<PaddedBuffer>& buffers,
                                   const nuts::InitConfig& init_cfg,
                                   const nuts::WarmupConfig& warmup_cfg,
                                   Stopper stop_all) {
  const std::size_t M = init_cfg.num_chains();
  const std::size_t D = init_cfg.dims();

  std::vector<float> mean_log_mass(D, 0.0f);
  std::vector<float> mean_mass(D, 0.0f);
  std::vector<float> scratch_mass(D, 0.0f);

  float mean_log_step = 0.0f;

  std::uint32_t min_iter = 0;

  auto probe_period =
      std::chrono::microseconds(warmup_cfg.probe_microseconds());

  auto next = std::chrono::steady_clock::now() + probe_period;
  std::vector<const AdaptSnapshot*> latest(M, nullptr);
  while (true) {
    std::fill(mean_log_mass.begin(), mean_log_mass.end(), 0.0f);
    mean_log_step = 0.0f;
    min_iter = std::numeric_limits<std::uint32_t>::max();
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

    float max_rel_mass = 0.0f;
    float max_rel_step = 0.0f;

    for (std::size_t m = 0; m < M; ++m) {
      const AdaptSnapshot& s = *latest[m];  //  buffers[m].val.read_latest();

      const float diff_mass = l2_rel_diff(std::span<const float>(s.mass),
                                          std::span<const float>(mean_mass));
      max_rel_mass = std::max(max_rel_mass, diff_mass);

      // Step comparison on log scale: |log_step_m - mean_log_step| /
      // |mean_log_step|
      double s_step = std::exp(s.log_step);
      double m_step = std::exp(mean_log_step);
      const float rel_step = (s_step - m_step) / m_step;
      max_rel_step = std::max(max_rel_step, rel_step);
    }

    // Optional progress print (single line).
    std::cout << '\r' << "min_iter=" << min_iter
              << "  max_rel_mass=" << max_rel_mass
              << "  max_rel_step=" << max_rel_step << std::flush;

    const bool enough_iters = (min_iter >= warmup_cfg.min_iter());
    const bool converged = enough_iters &&
                           max_rel_mass <= warmup_cfg.mass_converge_tol() &&
                           max_rel_step <= warmup_cfg.step_size_converge_tol();

    const bool hit_max = min_iter >= warmup_cfg.max_iter();

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
    next += probe_period;
  }
}

template <typename Adapter>
AdaptResult adapt(const nuts::InitConfig& init_cfg,
                  const nuts::WarmupConfig& warmup_cfg,
                  std::vector<Adapter>& adapters) {
  std::vector<PaddedBuffer> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());
  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains()));

  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    std::uint32_t chain_id = static_cast<std::uint32_t>(m);
    threads.emplace_back(AdaptWorker<Adapter>(
        chain_id, init_cfg, warmup_cfg, buffers[m], start_gate, adapters[m]));
  }
  auto stop_all = [&] {
    for (auto& t : threads) {
      t.request_stop();
    }
  };
  return controller_loop(buffers, init_cfg, warmup_cfg, stop_all);
}

// ********** EXAMPLE AFTER HERE **************

// this is a stub that will get replaced with WALNUTS
class MyAdapter {
 public:
  MyAdapter(std::size_t dim, std::uint64_t seed)
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
    ++iter_;
  }

  float log_step() const noexcept { return log_step_; }

  std::span<const float> log_mass() const noexcept { return log_mass_; }

  std::size_t iter() const noexcept { return iter_; }

 private:
  static std::vector<float> means(std::size_t dim) {
    std::vector<float> m(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      const float x = static_cast<float>(d + 1);
      m[d] = std::log(x * x);
    }
    return m;
  }

  std::size_t iter_ = 0;
  std::size_t dim_;
  std::mt19937_64 rng_;
  std::normal_distribution<float> z_;
  std::vector<float> log_mass_means_;
  std::vector<float> log_mass_;
  float log_step_;
};

int main() {
  std::cout << "Adaptation Monitoring Demo" << std::endl;

  uint64_t chains = 32;
  uint64_t dim = 100;
  auto init_cfg = nuts::InitConfigBuilder(chains, dim).build();

  auto warmup_cfg = nuts::WarmupConfigBuilder()
                        .min_max_iter(20, 2000)
                        .publish_stride(5)
                        .probe_microseconds(1000)
                        .mass_converge_tol(1.0)
                        .step_size_converge_tol(0.08)
                        .build();

  std::mt19937_64 rng(123456);
  std::vector<MyAdapter> adapters;
  adapters.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    adapters.emplace_back(MyAdapter(init_cfg.dims(), rng()));
  }

  AdaptResult res = adapt<MyAdapter>(init_cfg, warmup_cfg, adapters);

  const float mass_bar_norm = l2_norm(std::span<const float>(res.mass_bar));

  std::cout << "\nSHARED ADAPTED RESULT:  "
            << "stop_iter_min=" << res.stop_iter_min
            << "  step_bar=" << res.step_bar
            << "  ||mass_bar||=" << mass_bar_norm << '\n';

  std::cout << "\nPER CHAIN FINAL STATES:\n";
  for (std::size_t m = 0; m < adapters.size(); ++m) {
    std::cout << m << ")"
              << " iter = " << adapters[m].iter()
              << "  step = " << std::exp(adapters[m].log_step())
              << "  ||log_mass|| = "
              << l2_norm(std::span<const float>(adapters[m].log_mass()))
              << std::endl;
  }

  return 0;
}
