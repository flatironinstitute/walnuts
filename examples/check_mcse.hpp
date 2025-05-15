#ifndef WALNUTS_EXAMPLES_MCSE_CHECK
#define WALNUTS_EXAMPLES_MCSE_CHECK

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace walnuts::test {

namespace internal {
// Standard normal CDF via std::erf
inline double Phi(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Compute per‐trial failure probability p0 = P(|Z| > mcse_dev) for Z∼N(0,1)
inline double failure_prob(double mcse_dev) {
  return std::erfc(mcse_dev / std::sqrt(2.0));
}

// Approximate Binomial(K,p0) upper‐tail P(X ≥ n_fail) via normal‐approx
double binom_test(int K, int n_fail, double p0) {
  double mean = K * p0;
  double var = K * p0 * (1.0 - p0);
  if (var <= 0.0)
    return (n_fail > mean ? 0.0 : 1.0);
  double Z = (n_fail - mean) / std::sqrt(var);
  return 1.0 - Phi(Z); // one‐sided
}
} // namespace internal
// ——— Result struct ———
struct McseResult {
  int num_fails;  // total mean+variance failures
  double p_value_mean; // omnibus binomial p-value
  double p_value_vars;
  bool passed;    // num_fails ≤ max_fails
};

// ——— Updated check_mcse ———
template <typename S>
McseResult check_mcse(const Eigen::Matrix<S, -1, -1> &draws,
                      const Eigen::Matrix<S, -1, 1>& true_means,
                      const Eigen::Matrix<S, -1, 1>& true_m2s,
                      S tolerance = 1e-8, S mcse_deviation = 3.0,
                      int max_fails = -1) {
  const int D = draws.rows();
  const int N = draws.cols();
  // Default heuristic assumes
  S p_fail = internal::failure_prob(mcse_deviation);
  if (max_fails < 0) {
    S mean = D * p_fail;
    S var = D * p_fail * (1.0 - p_fail);
    S sd = std::sqrt(var);
    int m = static_cast<int>(std::ceil(mean + mcse_deviation * sd));
    max_fails = std::max(m + 1, 1);
    std::cout << "max_fails default = " << max_fails << "\n";
  }

  int num_mean_fails = 0;
  int num_m2_fails = 0;
  auto mean_v = draws.rowwise().mean().eval();
  auto m2_v = draws.array().square().rowwise().mean().eval();
  for (int d = 0; d < D; ++d) {
    // — sample mean & variance —
    auto true_mean = true_means(d);
    auto mean = mean_v(d);
    auto true_m2 = true_m2s(d);
    auto m2 = m2_v(d);
    // — mean test —
    S se_mean = std::sqrt(true_m2 / N);
    S se_mean_dev = se_mean * mcse_deviation;
    S mean_threshold = std::max(
        tolerance * 0.5 * (std::fabs(mean) + std::fabs(se_mean_dev)), 1e-14);
    if (std::fabs(mean - true_mean) - se_mean_dev > mean_threshold) {
      num_mean_fails++;
    }

    // — variance test —
    S se_m2 = std::sqrt(2.0 * m2 * m2 / (N - 1));
    S se_m2_dev = se_m2 * mcse_deviation;
    S m2_threshold = std::max(
        tolerance * 0.5 * (std::fabs(m2) + std::fabs(se_m2_dev)), 1e-14);
    if (std::fabs(m2 - true_m2) - se_m2_dev > m2_threshold) {
      num_m2_fails++;
    }
  }

  // — compute omnibus p-value —
  // here we treat each test (mean+var) separately → K = 2D
  double pval_means = internal::binom_test(D, num_mean_fails, p_fail);
  double pval_m2s = internal::binom_test(D, num_m2_fails, p_fail);

  const bool failed = (pval_means < 0.01 || pval_m2s < 0.01);
  if (failed) {
    std::cerr << "\n*************"
                 "\n** WARNING **"
                 "\n*************\n";
    if (pval_means < 0.01) {
      std::cout << "\tMean test failed " << num_mean_fails << " out of " << D << " times\n";
    }
    if (pval_m2s < 0.01) {
      std::cout << "\tsecond moment test failed " << num_m2_fails << " out of " << D << " times\n";
    }
  }

  std::cout << "Mean:\n\tfailures =  " << num_mean_fails << "\n";
  std::cout << "\tp-value  =  " << pval_means << "\n";
  std::cout << "second moment\n\tfailures =  " << num_m2_fails << "\n";
  std::cout << "\tp-value  =  " << pval_m2s << "\n";

  return McseResult{num_mean_fails + num_m2_fails, pval_means, pval_m2s, !failed};
}

} // namespace walnuts::test

#endif
