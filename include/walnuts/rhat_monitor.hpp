#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

class RhatMonitor {
 public:
  RhatMonitor(size_t num_chains, double rhat_tolerance)
      : num_chains_(num_chains),
        rhat_tolerance_(rhat_tolerance),
        mu_(num_chains),
        s_sq_(num_chains),
        updates_(num_chains),
        is_converged_(false) {
    for (size_t m = 0; m < num_chains_; ++m) {
      mu_[m].store(std::numeric_limits<double>::quiet_NaN(),
                   std::memory_order_relaxed);
      s_sq_[m].store(std::numeric_limits<double>::quiet_NaN(),
                     std::memory_order_relaxed);
      updates_[m].store(0, std::memory_order_relaxed);
    }
  }

  // sequential access per chain_id; concurrent across chain_ids
  void update(size_t chain_id, double mean, double variance) noexcept {
    mu_[chain_id].store(mean, std::memory_order_relaxed);
    s_sq_[chain_id].store(variance, std::memory_order_relaxed);
    updates_[chain_id].fetch_add(1, std::memory_order_relaxed);
  }

  bool is_converged() const noexcept {
    // should this and the store() in monitor be relaxed?
    return is_converged_.load(std::memory_order_acquire); // relaxed?
  }

  void monitor() {
    while (!is_converged_.load(std::memory_order_relaxed)) {
      if (rhat() < rhat_tolerance_) {
        is_converged_.store(true, std::memory_order_release);  // relaxed?
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

 private:
  template <typename T>
  static double mean(const std::vector<std::atomic<T>>& x) noexcept {
    const auto N = x.size();
    double s = 0.0;
    for (const auto& a : x) {
      s += a.load(std::memory_order_relaxed);
    }
    return s / N;
  }

  static double prim_variance(const std::vector<double>& x) noexcept {
    const auto N = x.size();  // caller ensures N >= 2
    double sum = 0;
    for (const auto& a : x) {
      sum += a;
    }
    double mean = sum / N;
    double sum_sq_diffs = 0;
    for (const auto& a : x) {
      double diff = a - mean;
      sum_sq_diffs += diff * diff;
    }
    return sum_sq_diffs / (N - 1);
  }  

  static double variance(const std::vector<std::atomic<double>>& x) {
    const auto N = x.size();
    if (N < 2) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> x_prim(N);
    for (int n = 0; n < N; ++n) {
      x_prim[n] = x[n].load(std::memory_order_relaxed);
    }
    return prim_variance(x_prim);
  }

  // weights by chain; N is arithmetic mean across chains
  double rhat() const noexcept {
    if (num_chains_ < 2) {
      return std::numeric_limits<double>::infinity();
    }
    const double N = mean(updates_);
    if (N < 2.0) {
      return std::numeric_limits<double>::infinity();
    }
    const double W = mean(s_sq_);
    if (!(W > 0.0)) {
      return std::numeric_limits<double>::infinity();
    }
    const double B_over_N = variance(mu_);
    const double Vhat = ((N - 1.0) / N) * W + B_over_N;
    return std::sqrt(Vhat / W);
  }

  const size_t num_chains_;
  const double rhat_tolerance_;
  std::vector<std::atomic<double>> mu_;
  std::vector<std::atomic<double>> s_sq_;
  std::vector<std::atomic<size_t>> updates_;
  std::atomic<bool> is_converged_;
};
