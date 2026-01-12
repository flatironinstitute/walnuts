// clang++ -std=c++20 -march=native -O3 -pthread -I ../include rhat_monitor.cpp -o rhat_monitor
// ./rhat_monitor

#include <array>
#include <atomic>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <walnuts/padded.hpp>

#if defined __has_attribute
#if __has_attribute(always_inline)
#ifndef WALNUTS_STRONG_INLINE
#define WALNUTS_STRONG_INLINE [[gnu::always_inline]] inline
#endif
#else
#define WALNUTS_STRONG_INLINE inline
#endif
#endif

#ifdef __APPLE__
#include <pthread.h>
WALNUTS_STRONG_INLINE void interactive_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // best
}
WALNUTS_STRONG_INLINE void initiated_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);  // next best
}
#else
WALNUTS_STRONG_INLINE void interactive_qos() {}
WALNUTS_STRONG_INLINE void initiated_qos() {}
#endif

double sum(const std::vector<double>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), 0.0, std::plus<>{},
                               std::identity());
}

std::size_t sum(const std::vector<std::size_t>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), std::size_t(0),
                               std::plus<>{}, std::identity());
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
    ChainStats init{std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(), 0u};
    data_.store(init, std::memory_order_relaxed);
  }

  // conservatie release/acquire vs. relaxed pattern
  void store(const ChainStats& p, std::memory_order mem_order = std::memory_order_release) noexcept {
    data_.store(p, mem_order);
  }

  ChainStats load(std::memory_order mem_order = std::memory_order_acquire) const noexcept {
    return data_.load(mem_order);
  }

 private:
  std::atomic<ChainStats> data_;
};

class ChainRecord {
 public:
  ChainRecord(std::size_t D, std::size_t Nmax)
      : D_(D), Nmax_(Nmax), n_(0), theta_(Nmax * D), logp_(Nmax) {}

  std::span<double> draw(std::size_t n) noexcept {
    return {theta_.data() + n * D_, D_};
  }

  std::span<const double> draw(std::size_t n) const noexcept {
    return {&theta_[n * D_], D_};
  }

  double& logp(std::size_t n) noexcept { return logp_[n]; }
  const double& logp(std::size_t n) const noexcept { return logp_[n]; }

  void commit() noexcept { ++n_; }

  std::size_t dims() const noexcept { return D_; }

  std::size_t num_draws() const noexcept { return n_; }

  double operator()(std::size_t n, std::size_t d) const noexcept {
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

 private:
  const std::size_t D_;
  const std::size_t Nmax_;
  std::size_t n_;
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
    return {static_cast<float>(mean()), static_cast<float>(sample_variance()),
            static_cast<unsigned int>(count())};
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
  ChainWorker(std::size_t draws_per_chain, Sampler& sampler,
	      ChainRecord& chain_record,
              AtomicChainStats& acs, std::latch& start_gate,
	      std::size_t yield_period = 1024)
      : draws_per_chain_(draws_per_chain),
        sampler_(sampler),
        chain_record_(chain_record), // sampler.dim(), draws_per_chain),
        acs_(acs),
        start_gate_(start_gate),
	yield_period_(yield_period) {}

  void operator()(std::stop_token st) {
    interactive_qos();
    start_gate_.arrive_and_wait();
    for (std::size_t iter = 0;
	 iter < draws_per_chain_ && !st.stop_requested();
         ++iter) {
      if (iter % yield_period_ == 0) {
        std::this_thread::yield();
      }
      auto draw = chain_record_.draw(iter);
      auto& lp = chain_record_.logp(iter);
      lp = sampler_.get().sample(draw);
      chain_record_.commit();
      logp_stats_.observe(lp);
      acs_.get().store(logp_stats_.sample_stats());
    }
  }

private:
  const std::size_t draws_per_chain_;
  std::reference_wrapper<Sampler> sampler_;
  ChainRecord& chain_record_;
  WelfordAccumulator logp_stats_;
  std::reference_wrapper<AtomicChainStats> acs_;
  std::latch& start_gate_;
  const std::size_t yield_period_;
};


template <typename Stopper>
static void controller_loop(std::vector<walnuts::Padded<AtomicChainStats>>& stats_by_chain,
                            double rhat_threshold, std::latch& start_gate,
                            std::size_t max_draws_per_chain,
                            Stopper stop_chains,
			    std::size_t& num_rhat_evals,
			    std::chrono::milliseconds eval_period = std::chrono::milliseconds{10}) {
  initiated_qos();
  num_rhat_evals = 0;
  const std::size_t M = stats_by_chain.size();
  std::vector<double> chain_means(M, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> chain_variances(M,
                                      std::numeric_limits<double>::quiet_NaN());
  std::vector<std::size_t> counts(M, 0);

  start_gate.wait();
  auto next = std::chrono::steady_clock::now() + eval_period;
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

    ++num_rhat_evals;

    // PLACEHOLDER FOR DEBUGGING---GET RID OF ALL std::cout FOR SERVER VERSION
    std::cout << std::setprecision(6) << std::fixed
	      << std::setw(8) << std::setfill(' ')
	      << '\r' // begin of currnet line
	      << "R hat " << r_hat
	      << "  #draws " << num_draws
	      << std::flush;
    
    if (r_hat <= rhat_threshold || num_draws == M * max_draws_per_chain) {
      break;
    }
    // we only need so many evaluations per second in a user-facing app
    std::this_thread::sleep_until(next);
    next += eval_period;
  }

  stop_chains();
}

// Sampler requires: { double sample(span<double> draw);  size_t dim(); }
template <typename Sampler>
std::vector<ChainRecord> sample(std::vector<Sampler>& samplers,
                                double rhat_threshold,
                                std::size_t max_draws_per_chain,
				std::size_t& num_rhat_evals) {
  std::size_t M = samplers.size();
  std::vector<walnuts::Padded<AtomicChainStats>> stats_by_chain(M);
  std::latch start_gate(M);
  std::vector<ChainRecord> chain_records;
  chain_records.reserve(M);
  std::vector<std::jthread> threads;
  threads.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    chain_records.emplace_back(samplers[m].dim(), max_draws_per_chain);
    threads.emplace_back(ChainWorker<Sampler>(max_draws_per_chain, samplers[m],
					      chain_records[m],
					      stats_by_chain[m].val, start_gate));
  }
  auto stop_all = [&] {
    for (auto& t : threads) {
      t.request_stop();
    }
  };
  controller_loop(stats_by_chain, rhat_threshold, start_gate,
                  max_draws_per_chain, stop_all, num_rhat_evals);
  return chain_records;
}

template <typename T, typename Out>
static inline void write_u32(Out& out, T x) {
  auto y = static_cast<std::uint32_t>(x);
  out.write(reinterpret_cast<const char*>(&y), sizeof(y));
}

template <typename Out>
static inline void write_f64(Out& out, double x) {
  auto bytes = reinterpret_cast<const char*>(&x);
  out.write(bytes, 8);
}

template <typename Out>
static inline void write_string(Out& out, const std::string& s) {
  write_u32(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

template <typename Out>
static void write_fixed_header(Out& out, std::size_t dim,
                               const std::vector<ChainRecord>& chains) {
  static constexpr std::string_view NAME = "WALNUTS";
  static constexpr std::uint32_t VERSION = 1;
  out.write(NAME.data(), NAME.size());
  write_u32(out, VERSION);
  write_u32(out, chains.size());
  write_u32(out, dim);
}

template <typename Out>
static void write_column_names(Out& out, std::size_t dim) {
  write_u32(out, dim + 1);
  write_string(out, "log_density");
  for (std::size_t d = 0; d < dim; ++d) {
    std::string param_name = "theta[" + std::to_string(d) + "]";
    write_string(out, param_name);
  }
}

template <typename Out>
static void write_chains(Out& out, std::size_t dim,
                         const std::vector<ChainRecord>& chains) {
  for (std::size_t chain_id = 0; chain_id < chains.size(); ++chain_id) {
    const ChainRecord& rec = chains[chain_id];
    const std::size_t N = rec.num_draws();
    for (std::size_t n = 0; n < N; ++n) {
      write_u32(out, chain_id);
      write_u32(out, n);
      write_f64(out, rec.logp(n));
      const auto row = rec.draw(n);
      out.write(reinterpret_cast<const char*>(row.data()),
		static_cast<std::streamsize>(row.size_bytes()));
    }
  }
}

// assumes little endian, 8-byte double
// assumes 8-byte double
static void write_binary(const std::string& path, std::size_t dim,
                         const std::vector<ChainRecord>& chains) {
  static_assert(std::endian::native == std::endian::little,
		"Binary format requires little-endian host");
  static_assert(sizeof(double) == 8, "Binary format requires 8-byte double");

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open file: " + path);
  }
  std::vector<char> filebuf(8u << 20); //  8 MB buffer bigger than usual
  out.rdbuf()->pubsetbuf(filebuf.data(), static_cast<std::streamsize>(filebuf.size()));

  std::ostringstream oss;
  write_fixed_header(oss, dim, chains);
  write_column_names(oss, dim);
  auto header = oss.str();
  out << header;
  std::size_t padding_to_8_byte_boundary = 8 - header.size() % 8;   
  for (std::size_t i = 0; i < padding_to_8_byte_boundary; ++i) {
    out << '\0';
  }

  write_chains(out, dim, chains);
  if (!out) {
    throw std::runtime_error("I/O error writing binary file");
  }
}

// ****************** EXAMPLE USE *******************************

class StandardNormalSampler {
 public:
  explicit StandardNormalSampler(unsigned int seed, std::size_t dim)
      : dim_(dim), engine_(seed), normal_dist_(0, 1) {}

  double sample(std::span<double> draw) noexcept {
    double lp = 0;
    for (double& x : draw) {
      x = normal_dist_(engine_);
      lp += -0.5 * x * x;  // unnomalized
    }
    std::this_thread::sleep_for(std::chrono::microseconds{500}); // sim Stan
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
  std::cout << "LOCK FREE: "
	    << ( std::atomic<ChainStats>().is_lock_free() ? "yes" : "***NO***")
	    << std::endl;

  const std::string out_path = "sample.mcmc";
  const std::size_t D = 100;
  std::size_t M = 64;
  const std::size_t N = 100000;
  double rhat_threshold = 1.0001;
  unsigned int seed = 1234;  // not reproducible because of threading!

  std::mt19937 rd(seed);
  std::vector<StandardNormalSampler> samplers;
  samplers.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    auto seed = rd();
    samplers.emplace_back(seed, D);
  }

  std::size_t num_rhat_evals;
  std::vector<ChainRecord> chain_records = sample(samplers, rhat_threshold, N, num_rhat_evals);
  std::cout << "num Rhat evals = " << num_rhat_evals;
  std::size_t rows = 0;
  for (std::size_t m = 0; m < chain_records.size(); ++m) {
    const auto& chain_record = chain_records[m];
    std::size_t N_m = chain_record.num_draws();
    std::vector<double> lps(N_m);
    for (std::size_t n = 0; n < N_m; ++n) {
      lps[n] = chain_record.logp(n);
    }
    rows += N_m;
    std::cout << "Chain " << m << "  count " << N_m
              << "  mean(logp) " << mean(lps)
              << "  sd(logp) [sample] " << std::sqrt(variance(lps)) << '\n';
  }
  std::cout << "Number of draws: " << rows << '\n';

  // uncomment to dump csv or binary .mcmc formats
  // write_binary(out_path, D, chain_records); // WARNING: up to N * D * 8 bytes
  // write_csv(out_path, D, chain_records);  // WARNING: even bigger
  // std::cout << "Wrote draws to " << out_path << '\n';

  return 0;
}
