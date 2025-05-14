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
double binom_omnibus_normal(int K, int n_fail, double p0) {
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
  int num_var_fails = 0;
  for (int d = 0; d < D; ++d) {
    // — sample mean & variance —
    S mean = draws.row(d).mean();
    S var = (draws.row(d).array() - mean).square().sum() / (N - 1);

    // — mean test —
    S se_mean = std::sqrt(var / N);
    S se_mean_dev = se_mean * mcse_deviation;
    S mean_threshold = std::max(
        tolerance * 0.5 * (std::fabs(mean) + std::fabs(se_mean_dev)), 1e-14);
    if (std::fabs(mean) - se_mean_dev > mean_threshold) {
      num_mean_fails++;
    }

    // — variance test —
    S se_var = std::sqrt(2.0 * var * var / (N - 1));
    S se_var_dev = se_var * mcse_deviation;
    S var_threshold = std::max(
        tolerance * 0.5 * (std::fabs(var - 1.0) + std::fabs(se_var_dev)),
        1e-14);
    if (std::fabs(var - 1.0) - se_var_dev > var_threshold) {
      num_var_fails++;
    }
  }

  // — compute omnibus p-value —
  // here we treat each test (mean+var) separately → K = 2D
  double pval_means = internal::binom_omnibus_normal(D, num_mean_fails, p_fail);
  double pval_vars = internal::binom_omnibus_normal(D, num_var_fails, p_fail);

  const bool failed = (pval_means < 0.01 || pval_vars < 0.01);
  if (failed) {
    std::cerr << "\n*************"
                 "\n** WARNING **"
                 "\n*************\n";
    if (pval_means < 0.01) {
      std::cout << "\tMean test failed " << num_mean_fails << " out of " << D << " times\n";
    }
    if (pval_vars < 0.01) {
      std::cout << "\tVariance test failed " << num_var_fails << " out of " << D << " times\n";
    }
  }

  std::cout << "Mean:\n\tfailures =  " << num_mean_fails << "\n";
  std::cout << "\tp-value  =  " << pval_means << "\n";
  std::cout << "Variance\n\tfailures =  " << num_var_fails << "\n";
  std::cout << "\tp-value  =  " << pval_vars << "\n";

  return McseResult{num_mean_fails + num_var_fails, pval_means, pval_vars, !failed};
}

} // namespace walnuts::test

#endif