#include <walnuts/nuts.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

double total_time = 0.0;
int count = 0;

template <typename T>
void standard_normal_logp_grad(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
                               S &logp,
                               Eigen::Matrix<T, Eigen::Dynamic, 1> &grad) {
  auto start = std::chrono::high_resolution_clock::now();
  logp = -0.5 * x.dot(x);
  grad = -x;
  auto end = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end - start).count();
  ++count;
}

/**
 * Check that each row of `draws` has mean ~ 0 and variance ~ 1 within
 *  Monte Carlo standard error bounds.
 *
 * @param draws A D×N matrix whose d-th row is the N samples for dimension d.
 *
 * @param tolerance A small positive value to used as basis for an adjusted relative threshold.
 *
 * @param mcse_deviation magnitude of Monte Carlo standard errors (SE) to allow before failing
 *
 * @param max_fails Max number of failures before the overall test returns a failure
 *
 * @returns 0 if the number of failing dimensions is < max_fails, otherwise 1.
 */
int check_mcse(const Eigen::Matrix<S, -1, -1>& draws,
               double tolerance = 1e-8,
               double mcse_deviation = 3.0,
               int max_fails = 10) {
  int num_mean_fails = 0;
  int num_var_fails = 0;
  const int D = draws.rows();
  const int N = draws.cols();

  for (int d = 0; d < D; ++d) {
    S mean = draws.row(d).mean();
    S var  = (draws.row(d).array() - mean).square().sum() / (N - 1);

    S se_mean    = std::sqrt(var / N);
    S se_mean_dev= se_mean * mcse_deviation;
    auto mean_threshold
      = std::max(tolerance * 0.5 * (std::fabs(mean) + std::fabs(se_mean_dev)), 1e-14);

    S abs_mean = std::fabs(mean);
    if (abs_mean > se_mean_dev + mean_threshold) {
      S diff_from_mcse      = abs_mean - se_mean_dev;
      S diff_from_total_tol = abs_mean - (se_mean_dev + mean_threshold);

      std::cerr
        << "\n[MCSE FAILURE] Dimension " << d << " (mean test)\n"
        << "  Sample mean                  = " << mean << "\n"
        << "  abs(mean)                    = " << abs_mean << "\n"
        << "  MCSE                         = " << se_mean << "\n"
        << "  adjusted relative threshold  = " << mean_threshold << "\n"
        << "  mcse_deviation               = " << mcse_deviation << "\n"
        << "  MCSE * deviation + threshold = " << se_mean_dev + mean_threshold << "\n"
        << "  Exceeded total bound by      = " << diff_from_total_tol << "\n";
      num_mean_fails++;
    }

    S se_var     = std::sqrt(2.0 * var*var / (N - 1));
    S se_var_dev = se_var * mcse_deviation;
    auto var_threshold
      = std::max(tolerance * 0.5 * (std::fabs(var - 1.0) + std::fabs(se_var_dev)), 1e-14);

    S abs_var_diff = std::fabs(var - 1.0);
    if (abs_var_diff > se_var_dev + var_threshold) {
      S diff_from_mcse      = abs_var_diff - se_var_dev;
      S diff_from_total_tol = abs_var_diff - (se_var_dev + var_threshold);

      std::cerr
        << "\n[MCSE FAILURE] Dimension " << d << " (var test)\n"
        << "  Sample var                    = " << var << "\n"
        << "  abs(var - 1)                  = " << abs_var_diff << "\n"
        << "  MCSE                          = " << se_var << "\n"
        << "  adjusted relative threshold   = " << var_threshold << "\n"
        << "  mcse_deviation                = " << mcse_deviation << "\n"
        << "  MCSE * deviation + threshold  = " << se_var_dev + var_threshold << "\n"
        << "  Exceeded total bound by       = " << diff_from_total_tol << "\n";
      num_var_fails++;
    }
  }
  std::cout << "Mean failures: " << num_mean_fails << "\n";
  std::cout << "Var failures:  " << num_var_fails << "\n";
  int num_fails = num_mean_fails + num_var_fails;
  if (num_fails >= max_fails) {
    std::cerr << "\n*** ABORT: " << num_fails
              << " dimensions outside Monte Carlo error bounds. ***\n";
    return 1;
  } else {
    return 0;
  }
}

int main() {
  int init_seed = 333456;
  int seed = 763545;
  int D = 200;
  int N = 5000;
  S step_size = 0.025;
  int max_depth = 10;
  VectorS inv_mass = VectorS::Ones(D);
  MatrixS draws(D, N);
  std::cout << "D = " << D << ";  N = " << N << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth << std::endl;

  std::mt19937 generator(init_seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts::nuts(generator, standard_normal_logp_grad<S>, inv_mass, step_size,
             max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time =
      std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "    total time: " << global_total_time << "s" << std::endl;
  std::cout << "logp_grad time: " << total_time << "s" << std::endl;
  std::cout << "logp_grad fraction: " << total_time / global_total_time
            << std::endl;
  std::cout << "        logp_grad calls: " << count << std::endl;
  std::cout << "        time per call: " << total_time / count << "s"
            << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < std::min(D, 10); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (D > 10) {
    std::cout << "... elided " << (D - 10) << " dimensions ..." << std::endl;
  }
  return check_mcse(draws);
}
