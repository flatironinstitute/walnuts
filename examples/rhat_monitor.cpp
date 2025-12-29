// clang++ -std=c++20 -O3 -pthread rhat_monitor.cpp -o rhat_monitor
// ./rhat_monitor

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define VERY_INLINE __attribute__((always_inline)) inline
#else
#define VERY_INLINE inline
#endif

#ifdef __APPLE__
#include <pthread.h>
VERY_INLINE void interactive_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // best
}
VERY_INLINE void initiated_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);  // next best
}
#else
VERY_INLINE void interactive_qos() {}
VERY_INLINE void initiated_qos() {}
#endif

constexpr std::size_t DI_SIZE =
  std::hardware_destructive_interference_size > 0
  ? std::hardware_destructive_interference_size
  : 128;

double sum(const std::vector<double>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), 0.0, std::plus<>{},
                               std::identity());
}

std::size_t sum(const std::vector<std::size_t>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), std::size_t(0), std::plus<>{},
                               std::identity());
}

double mean(const std::vector<double>& xs) noexcept {
  return sum(xs) / xs.size();
}

double variance(const std::vector<double>& xs) noexcept {
  std::size_t N = xs.size();
  if (N < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double mean_xs = mean(xs);
  double sum = std::transform_reduce(xs.begin(), xs.end(), 0.0, std::plus<>{},
                                     [mean_xs](double x) {
                                       double diff = x - mean_xs;
                                       return diff * diff;
                                     });
  return sum / (N - 1);
}

struct ChainStats {
  // downsized to ensure atomic<ChainStats> is lock free
  float sample_mean;
  float sample_var;
  unsigned int count;
};

class AtomicChainStats {
 public:
  AtomicChainStats() noexcept {
    ChainStats init{std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(), 0u};
    data_.store(init, std::memory_order_relaxed);
  }

  void store(const ChainStats& p) noexcept {
    data_.store(p, std::memory_order_relaxed); //  // release);
  }

  ChainStats load() const noexcept {
    return data_.load(std::memory_order_relaxed); // acquire);
  }

 private:
  std::atomic<ChainStats> data_;
};

struct alignas(DI_SIZE) PaddedChainStats {
  static constexpr std::size_t PAD_SIZE =
    sizeof(AtomicChainStats) < DI_SIZE ? DI_SIZE - sizeof(AtomicChainStats) : 0;

  AtomicChainStats val;
  std::array<std::byte, PAD_SIZE> pad{};
};


class ChainRecord {
 public:
  ChainRecord(std::size_t D, std::size_t Nmax)
      : D_(D), Nmax_(Nmax) {
    theta_.reserve(Nmax * D);
    logp_.reserve(Nmax);
  }

  std::vector<double>& draws() noexcept { return theta_; }

  std::size_t dims() const noexcept { return D_; }

  std::size_t num_draws() const noexcept { return logp_.size(); }

  inline double logp(std::size_t n) const { return logp_[n]; }

  inline double operator()(std::size_t n, std::size_t d) const {
    return theta_[n * D_ + d];
  }

  void write_csv(std::ostream& out, std::size_t chain_id) const {
    auto dim = dims();
    for (std::size_t n = 0; n < num_draws(); ++n) {
      out << chain_id << ',' << n << ',' << logp_[n];
      for (std::size_t d = 0; d < dim; ++d) {
        out << ',' << operator()(n, d);
      }
      out << '\n';
    }
  }

  void append_logp(double logp) { logp_.push_back(logp); }

 private:
  std::size_t D_;
  std::size_t Nmax_;
  std::vector<double> theta_;
  std::vector<double> logp_;
};

static void write_csv_header(std::ofstream& out, std::size_t dim) {
  out << "chain,iteration,log_density";
  for (std::size_t d = 0; d < dim; ++d) {
    out << ",theta[" << d << "]";
  }
  out << '\n';
}

static void write_csv(const std::string& path, std::size_t dim,
                      const std::vector<ChainRecord>& chain_records) {
  std::ofstream out(path, std::ios::binary);  // binary for Windows consistency
  if (!out) {
    throw std::runtime_error("could not open file: " + path);
  }
  write_csv_header(out, dim);
  for (std::size_t i = 0; i < chain_records.size(); ++i) {
    chain_records[i].write_csv(out, i);
  }
}

class WelfordAccumulator {
 public:
  WelfordAccumulator() : n_(0), mean_(0.0), M2_(0.0) {}

  void observe(double x) {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / static_cast<double>(n_);
    const double delta2 = x - mean_;
    M2_ += delta * delta2;
  }

  std::size_t count() const { return n_; }

  double mean() const { return mean_; }

  double sample_variance() const {
    return n_ > 1 ? (M2_ / static_cast<double>(n_ - 1))
                  : std::numeric_limits<double>::quiet_NaN();
  }

  ChainStats sample_stats() {
    return {
        static_cast<float>(mean()),
	static_cast<float>(sample_variance()),
	static_cast<unsigned int>(count())
    };
  }

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

template <class Sampler>
class ChainWorker {
 public:
  ChainWorker(std::size_t draws_per_chain,
              Sampler& sampler, AtomicChainStats& acs, std::latch& start_gate)
      : draws_per_chain_(draws_per_chain),
        sampler_(sampler),
        chain_record_(sampler.dim(), draws_per_chain),
        acs_(acs),
        start_gate_(start_gate) {}

  void operator()(std::stop_token st) {
    static constexpr std::size_t YIELD_PERIOD = 64;
    interactive_qos();
    start_gate_.arrive_and_wait();
    for (std::size_t iter = 0; iter < draws_per_chain_ && !st.stop_requested(); ++iter) {
      if (iter % YIELD_PERIOD == 0) {
        std::this_thread::yield();
      }
      double logp = sampler_.sample(chain_record_.draws());
      chain_record_.append_logp(logp);
      logp_stats_.observe(logp);
      acs_.store(logp_stats_.sample_stats());
    }
  }

  ChainRecord take_record() && { return std::move(chain_record_); }

 private:
  std::size_t draws_per_chain_;
  Sampler& sampler_;
  ChainRecord chain_record_;
  WelfordAccumulator logp_stats_;
  AtomicChainStats& acs_;
  std::latch& start_gate_;
};

template <class Sampler>
class ChainRunner {
 public:
  ChainRunner(std::size_t draws_per_chain,
              Sampler& sampler,
              AtomicChainStats& acs,
              std::latch& start_gate)
      : worker_(draws_per_chain, sampler, acs, start_gate),
        thread_(std::ref(worker_)) {}

  ChainRunner(const ChainRunner&) = delete;
  ChainRunner& operator=(const ChainRunner&) = delete;
  ChainRunner(ChainRunner&&) noexcept = default;
  ChainRunner& operator=(ChainRunner&&) noexcept = default;

  void request_stop() { thread_.request_stop(); }

  void join() { thread_.join(); }
	   
  ChainRecord take_record() {
    return std::move(worker_).take_record();
  }

 private:
  ChainWorker<Sampler> worker_;
  std::jthread thread_;
};


void debug_print(double variance_of_means, double mean_of_variances,
                 double num_draws, double r_hat,
                 const std::vector<std::size_t>& counts) {
  auto M = counts.size();
  std::cout << "RHAT: " << r_hat << "  NUM_DRAWS: " << num_draws
            << "  COUNTS: ";
  for (std::size_t m = 0; m < M; ++m) {
    if (m > 0) {
      std::cout << ", ";
    }
    std::cout << counts[m];
  }
  std::cout << std::endl;
}

template <typename Stopper>
static void controller_loop(std::vector<PaddedChainStats>& stats_by_chain,
                            double rhat_threshold, std::latch& start_gate,
                            std::size_t max_draws_per_chain,
                            Stopper stop_chains) {
  static constexpr auto PERIOD = std::chrono::milliseconds{10};
  
  initiated_qos();
  start_gate.wait();
  const std::size_t M = stats_by_chain.size();
  std::vector<double> chain_means(M, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> chain_variances(M,
                                      std::numeric_limits<double>::quiet_NaN());
  std::vector<std::size_t> counts(M, 0);
  auto next = std::chrono::steady_clock::now() + PERIOD;
  while (true) {
    for (std::size_t m = 0; m < M; ++m) {
      ChainStats u = stats_by_chain[m].val.load();
      chain_means[m] = u.sample_mean;
      chain_variances[m] = u.sample_var;
      counts[m] = u.count;
    }
    double variance_of_means = variance(chain_means);
    double mean_of_variances = mean(chain_variances);
    double r_hat = std::sqrt(1 + variance_of_means / mean_of_variances);
    std::size_t num_draws = sum(counts);

    debug_print(variance_of_means, mean_of_variances, num_draws, r_hat, counts);

    if (r_hat <= rhat_threshold || num_draws == M * max_draws_per_chain) {
      break;
    }
    std::this_thread::sleep_until(next);
    next += PERIOD;
  }
  stop_chains();
}

// Sampler requires: { double sample(vector<double>& draw);  size_t dim(); }
template <typename Sampler>
std::vector<ChainRecord> sample(std::vector<Sampler>& samplers,
                                double rhat_threshold,
                                std::size_t max_draws_per_chain) {
  std::size_t M = samplers.size();
  std::vector<PaddedChainStats> stats_by_chain(M);
  std::latch start_gate(M);

  std::vector<ChainRunner<Sampler>> runners;
  runners.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    runners.emplace_back(max_draws_per_chain, samplers[m], stats_by_chain[m].val, start_gate);
  }

  auto stop_all = [&] { for (auto& r : runners) r.request_stop(); };
  controller_loop(stats_by_chain, rhat_threshold, start_gate, max_draws_per_chain, stop_all);
  for (auto& r : runners) r.join(); // avoid race before taking records
  
  std::vector<ChainRecord> chain_records;
  chain_records.reserve(M);
  for (auto& r : runners) chain_records.emplace_back(r.take_record());
  return chain_records;
}

// ****************** EXAMPLE USAGE AFTER HERE ************************

class StandardNormalSampler {
 public:
  explicit StandardNormalSampler(unsigned int seed, std::size_t dim)
      : dim_(dim), engine_(seed), normal_dist_(0, 1) {}

  double sample(std::vector<double>& draw) noexcept {
    double lp = 0;
    for (std::size_t i = 0; i < dim_; ++i) {
      double x = normal_dist_(engine_);
      draw.push_back(x);
      lp += -0.5 * x * x;  // unnomalized
    }
    return lp;
  }

  std::size_t dim() const noexcept { return dim_; }

 private:
  std::size_t dim_;
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_dist_;
};

int main() {
  std::atomic<ChainStats> test_chain_stats;
  std::cout << "atomic<ChainStats>().is_lock_free() = " << test_chain_stats.is_lock_free() << std::endl; 

  
  const std::string csv_path = "samples.csv";
  const std::size_t D = 100;
  std::size_t M = 64;
  const std::size_t N = 100000;
  double rhat_threshold = 1.00001;

  std::random_device rd;
  std::vector<StandardNormalSampler> samplers;
  samplers.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    auto seed = rd();
    samplers.emplace_back(seed, D);
  }

  std::vector<ChainRecord> chain_records = sample(samplers, rhat_threshold, N);
  std::size_t rows = 0;
  for (std::size_t m = 0; m < chain_records.size(); ++m) {
    const auto& chain_record = chain_records[m];
    std::size_t N_m = chain_record.num_draws();
    std::vector<double> lps(N_m);
    for (std::size_t n = 0; n < N_m; ++n) {
      lps[n] = chain_record.logp(n);
    }
    rows += N_m;
    std::cout << "Chain " << m << "  count=" << N_m
              << "  Final: mean(logp)=" << mean(lps)
              << "  var(logp) [sample]=" << variance(lps) << '\n';
  }
  std::cout << "Number of draws: " << rows << '\n';

  // UNCOMMENT TO DUMP CSV
  // write_csv(csv_path, D, chain_records);
  // std::cout << "Wrote draws to " << csv_path << '\n';

  return 0;
}
