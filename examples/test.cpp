#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>

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
 * Compute the joint LRT statistic and an approximate p-value
 * (via the Wilson–Hilferty normal approximation) for
 * testing that each row of `draws` comes from N(0,1).
 *
 * @param draws  D×N matrix of samples (one row per dimension).
 * @return       pair(statistic, p_value_approx).
 */
std::pair<double,double> joint_lrt_test(
  const Eigen::MatrixXd &draws)
{
const int D = draws.rows();
const int N = draws.cols();

// 1) compute the LRT statistic
double lrt_stat = 0.0;
for (int d = 0; d < D; ++d) {
  // sample mean
  double mean = draws.row(d).mean();
  // sum of squared deviations
  double sumsq = (draws.row(d).array() - mean).square().sum();
  // MLE variance uses 1/N
  double sigma2 = sumsq / N;

  // per‐dimension contribution:
  //   N*mean^2 + N*(sigma2 - log(sigma2) - 1)
  lrt_stat += N * mean*mean
            + N * (sigma2 - std::log(sigma2) - 1.0);
}

// 2) approximate p‐value for χ²_{2D} via Wilson–Hilferty:
//    Z ≈ [ (lrt_stat/df)^{1/3} - (1 - 2/(9df)) ] / sqrt(2/(9df))
//    p ≈ 1 - Φ(Z)
int df = 2*D;
double y = lrt_stat / df;
double z = (std::pow(y, 1.0/3.0)
            - (1.0 - 2.0/(9.0*df)))
         / std::sqrt(2.0/(9.0*df));
double p_approx = 0.5 * std::erfc(z / std::sqrt(2.0));

return {lrt_stat, p_approx};
}


template <bool W, typename G>
void test_nuts(const VectorS &theta_init, G &generator, int D, int N,
               S step_size, S max_depth, S max_error, const VectorS &inv_mass) {
  MatrixS draws(D, N);
  std::cout << std::endl
            << "D = " << D << ";  N = " << N << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth
            << ";  WALNUTS = " << (W ? "true" : "false") << std::endl;

  auto global_start = std::chrono::high_resolution_clock::now();
  if constexpr (W) {
    walnuts::walnuts(generator, standard_normal_logp_grad<S>, inv_mass,
                     step_size, max_depth, max_error, theta_init, draws);
  } else {
    nuts::nuts(generator, standard_normal_logp_grad<S>, inv_mass, step_size,
               max_depth, theta_init, draws);
  }
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

  for (int d = 0; d < std::min(D, 5); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (D > 10) {
    std::cout << "... elided " << (D - 5) << " dimensions ..." << std::endl;
  }
  auto lrt_val = joint_lrt_test(draws);
  std::cout << "LRT stat = " << stat << ", approx p‐value = " << pval << "\n";
}

int main() {
  int seed = 763545;
  int D = 200;
  int N = 5000;
  S step_size = 0.1;
  int max_depth = 10;
  S max_error = 0.05;
  VectorS inv_mass = VectorS::Ones(D);

  std::mt19937 generator(seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  test_nuts<false>(theta_init, generator, D, N, step_size, max_depth, max_error,
                   inv_mass);
  test_nuts<true>(theta_init, generator, D, N, step_size, max_depth, max_error,
                  inv_mass);

  return 0;
}
