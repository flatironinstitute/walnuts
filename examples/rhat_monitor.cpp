// TO RUN
// clang++ -std=c++20 -O3 rhat_monitor.cpp -o rhat_monitor
// ./rhat_monitor

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <latch>
#include <random>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef __APPLE__
#include <pthread.h>
[[gnu::always_inline]] inline void interactive_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); // best
}
[[gnu::always_inline]] inline void initiated_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0); // next best
}
#else
[[gnu::always_inline]] inline void interactive_qos() {}
[[gnu::always_inline]] inline void initiated_qos() {}
#endif

double sum(const std::vector<double> &xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), 0.0, std::plus<>{},
                               [](double x) { return x; });
}

double mean(const std::vector<double> &xs) noexcept {
  return sum(xs) / xs.size();
}

double variance(const std::vector<double> &xs) noexcept {
  std::size_t N = xs.size();
  if (N < 2)
    return std::numeric_limits<double>::quiet_NaN();
  double mean_xs = mean(xs);
  double sum = std::transform_reduce(xs.begin(), xs.end(), 0.0, std::plus<>{},
                                     [mean_xs](double x) {
                                       double diff = x - mean_xs;
                                       return diff * diff;
                                     });
  return sum / (N - 1);
}

struct SampleStats {
  std::size_t count;
  double sample_mean;
  double sample_var;
};

class Sample {
public:
  Sample(std::size_t chain_id, std::size_t D, std::size_t Nmax)
      : chain_id_(chain_id), D_(D), Nmax_(Nmax) {
    theta_.reserve(Nmax * D);
    logp_.reserve(Nmax);
  }

  std::size_t num_draws() const noexcept { return logp_.size(); }

  void write_csv(std::ostream &out, std::size_t dim) const {
    for (std::size_t n = 0; n < num_draws(); ++n) {
      out << chain_id_ << ',' << n << ',' << logp_[n];
      for (std::size_t d = 0; d < dim; ++d)
        out << ',' << operator()(n, d);
      out << '\n';
    }
  }

  void append_draw(std::size_t iteration, std::size_t chain_id, double logp,
                   std::vector<double> &&draw) {
    logp_.emplace_back(logp);
    theta_.insert(theta_.end(), draw.begin(), draw.end());
  }

  double mean_logp() const noexcept { return mean(logp_); }

  double variance_logp() const noexcept { return variance(logp_); }

  inline double operator()(std::size_t n, std::size_t d) const {
    return theta_[n * D_ + d];
  }

private:
  std::size_t chain_id_;
  std::size_t D_;
  std::size_t Nmax_;
  std::vector<double> theta_;
  std::vector<double> logp_;
};

// see https://rigtorp.se/ringbuffer/
template <class T, std::size_t Capacity>
class alignas(std::hardware_destructive_interference_size) RingBuffer {
public:
  explicit RingBuffer() : data_(Capacity) {}

  template <class... Args> bool emplace(Args &&...args) noexcept {
    auto write_idx = write_idx_.load(std::memory_order_relaxed);
    auto next = write_idx + 1;
    if (next == Capacity)
      next = 0;
    if (next == read_idx_.load(std::memory_order_acquire))
      return false;
    data_[write_idx] = T(std::forward<Args>(args)...);
    write_idx_.store(next, std::memory_order_release);
    return true;
  }

  bool pop(T &out) noexcept {
    auto read_idx = read_idx_.load(std::memory_order_relaxed);
    if (read_idx == write_idx_.load(std::memory_order_acquire))
      return false;
    out = std::move(data_[read_idx]);
    auto next = read_idx + 1;
    if (next == Capacity)
      next = 0;
    read_idx_.store(next, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const noexcept { return Capacity; }

private:
  std::vector<T> data_;
  alignas(std::hardware_destructive_interference_size)
      std::atomic<std::size_t> read_idx_{0};
  alignas(std::hardware_destructive_interference_size)
      std::atomic<std::size_t> write_idx_{0};
};

constexpr std::size_t RING_CAPACITY = 8;

using Queue = RingBuffer<SampleStats, RING_CAPACITY>;

static void write_csv_header(std::ofstream &out, std::size_t dim) {
  out << "chain,iteration,log_density";
  for (std::size_t d = 0; d < dim; ++d)
    out << ",theta[" << d << "]";
  out << '\n';
}

static void write_csv(const std::string &path, std::size_t dim,
                      const std::vector<Sample> &samples) {
  std::ofstream out(path, std::ios::binary); // binary for Windows consistency
  if (!out)
    throw std::runtime_error("could not open file: " + path);
  write_csv_header(out, dim);
  for (const auto &sample : samples)
    sample.write_csv(out, dim);
}

class WelfordAccumulator {
public:
  WelfordAccumulator() : n_(0), mean_(0.0), M2_(0.0) {}

  void push(double x) {
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

template <class Sampler> class ChainTask {
public:
  ChainTask(std::size_t chain_id, std::size_t draws_per_chain, Sampler &sampler,
            Queue &q, std::latch &start_gate)
      : chain_id_(chain_id), draws_per_chain_(draws_per_chain),
        sampler_(sampler), sample_(chain_id, sampler.dim(), draws_per_chain),
        q_(q), start_gate_(start_gate) {}

  void operator()(std::stop_token st) {
    initiated_qos();
    start_gate_.arrive_and_wait();
    for (std::size_t iter = 0; iter < draws_per_chain_; ++iter) {
      auto [logp, theta] = sampler_();
      logp_stats_.push(logp);
      sample_.append_draw(iter, chain_id_, logp, std::move(theta));
      q_.emplace(logp_stats_.count(), logp_stats_.mean(),
                 logp_stats_.sample_variance());
      if (st.stop_requested())
        break;
    }
  }

  const Sample &sample() const { return sample_; }

  Sample &&take_sample() { return std::move(sample_); }

  double mean_logp() const { return logp_stats_.mean(); }

  double sample_variance_logp() const { return logp_stats_.sample_variance(); }

  std::size_t count_logp() const { return logp_stats_.count(); }

  const WelfordAccumulator &logp_stats() const { return logp_stats_; }

private:
  std::size_t chain_id_;
  std::size_t draws_per_chain_;
  Sampler &sampler_;
  Sample sample_;
  WelfordAccumulator logp_stats_;
  Queue &q_;
  std::latch &start_gate_;
};

static void controller_loop(std::stop_token st, std::vector<Queue> &queues,
                            std::vector<std::jthread> &workers,
                            double rhat_threshold, std::latch &start_gate) {
  start_gate.wait();
  const std::size_t M = static_cast<std::size_t>(queues.size());
  std::vector<double> chain_means(M, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> chain_variances(M,
                                      std::numeric_limits<double>::quiet_NaN());
  while (!st.stop_requested()) {
    for (std::size_t m = 0; m < M; ++m) {
      SampleStats u;
      while (queues[m].pop(u)) {
        chain_means[m] = u.sample_mean;
        chain_variances[m] = u.sample_var;
      }
    }
    double variance_of_means = variance(chain_means);
    double mean_of_variances = mean(chain_variances);
    double r_hat = std::sqrt(1 + variance_of_means / mean_of_variances);
    // print is just for debugging/demonstrating runs
    std::cout << "variance of means=" << variance_of_means
              << "; mean of variances=" << mean_of_variances
              << "; r_hat = " << r_hat << '\n';
    if (r_hat <= rhat_threshold) {
      for (auto &w : workers)
        w.request_stop();
      break;
    }
  }
}

template <typename Sampler>
std::vector<Sample> sample(std::vector<Sampler> &samplers,
                           double rhat_threshold,
                           std::size_t max_draws_per_chain) {
  using namespace std::chrono_literals;
  using Task = ChainTask<Sampler>;

  const std::size_t M = samplers.size();
  std::vector<Queue> queues(M);
  std::latch start_gate(M);
  std::vector<Task> tasks;
  tasks.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    tasks.emplace_back(m, max_draws_per_chain, samplers[m], queues[m],
                       start_gate);
  }
  { // scope for jthread RAII control
    std::vector<std::jthread> workers;
    workers.reserve(M);
    for (std::size_t m = 0; m < M; ++m) {
      workers.emplace_back(std::ref(tasks[m]));
    }
    std::jthread controller([&](std::stop_token st) {
      interactive_qos();
      controller_loop(st, queues, workers, rhat_threshold, start_gate);
    });
    // if we don't join here, don't get print updates of convergence
    for (auto& w : workers)
      if (w.joinable())
	w.join();
  }
  std::vector<Sample> samples;
  samples.reserve(M);
  for (auto &task : tasks) {
    samples.emplace_back(task.take_sample());
  }
  return samples;
}

// ****************** EXAMPLE USAGE AFTER HERE ************************

// a simple example---not part of API
class StandardNormalSampler {
public:
  explicit StandardNormalSampler(unsigned int seed, std::size_t dim)
      : dim_(dim), engine_(seed), normal_dist_(0.0, 1.0) {}

  std::tuple<double, std::vector<double>> operator()() noexcept {
    std::vector<double> draws(dim_);
    double log_density = 0.0;
    for (std::size_t i = 0; i < dim_; ++i) {
      double x = normal_dist_(engine_);
      draws[i] = x;
      log_density += -0.5 * x * x; // unnormalized
    }
    return {log_density, std::move(draws)};
  }

  std::size_t dim() const noexcept { return dim_; }

private:
  std::size_t dim_;
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_dist_;
};

int main() {
  const std::string csv_path = "samples.csv";
  const std::size_t D = 8;
  std::size_t M = 16;
  const std::size_t N = 50000;
  double rhat_threshold = 1.001;

  std::random_device rd;
  std::vector<StandardNormalSampler> samplers;
  samplers.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    unsigned int seed = rd();
    samplers.emplace_back(seed, D);
  }

  std::vector<Sample> samples = sample(samplers, rhat_threshold, N);
  std::size_t rows = 0;
  for (std::size_t m = 0; m < samples.size(); ++m) {
    const auto &sample = samples[m];
    rows += sample.num_draws();
    std::cout << "Chain " << m << "  count=" << sample.num_draws()
              << "  Final: mean(logp)=" << sample.mean_logp()
              << "  var(logp) [sample]=" << sample.variance_logp()
              << '\n';
  }
  std::cout << "Total rows: " << rows << '\n';

  write_csv(csv_path, D, samples);
  std::cout << "Wrote draws to " << csv_path << '\n';

  return 0;
}
