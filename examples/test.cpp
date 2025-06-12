#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include <walnuts/adaptive_walnuts.hpp>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

enum class Sampler { Nuts, Walnuts };

double total_time = 0.0;
int count = 0;

void standard_normal_logp_grad(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                               S& logp,
                               Eigen::Matrix<S, Eigen::Dynamic, 1>& grad) {
  auto start = std::chrono::high_resolution_clock::now();
  logp = -0.5 * x.dot(x);
  grad = -x;
  auto end = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end - start).count();
  ++count;
}

void summarize(const MatrixS& draws) {
  int N = draws.cols();
  int D = draws.rows();
  for (int d = 0; d < std::min(D, 5); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (D > 5) {
    std::cout << "... elided " << (D - 5) << " dimensions ..." << std::endl;
  }
}

template <typename RNG>
void test_adaptive_walnuts(VectorS theta_init, RNG& rng, int D, int N,
			   int max_nuts_depth, S log_max_error) {
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 5.0;
  double mass_iteration_offset = 4.0;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset);

  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_iteration_offset = 4.0;
  double learning_rate = 0.95;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
				 step_iteration_offset, learning_rate,
				 decay_rate);

  int max_step_depth = 5;
  nuts::WalnutsConfig walnuts_cfg(log_max_error, max_nuts_depth,
				  max_step_depth);

  nuts::AdaptiveWalnuts sample(rng, standard_normal_logp_grad,
			       theta_init, std::move(mass_cfg),
			       std::move(step_cfg), std::move(walnuts_cfg));

  std::cout << "\nTEST ADAPTIVE WALNUTS" << std::endl;
}

template <typename RNG>
void test_walnuts_iter(VectorS theta_init, RNG& rng, int D, int N,
		       S macro_step_size, int max_nuts_depth, S log_max_error,
		       VectorS inv_mass) {
  std::cout << "\nWALNUTS ITERATOR" << std::endl;
  nuts::WalnutsSampler sample(rng, standard_normal_logp_grad, theta_init,
			      inv_mass, macro_step_size, max_nuts_depth,
			      log_max_error);
  MatrixS draws(D, N);
  for (int n = 0; n < N; ++n) {
    draws.col(n) = sample();
  }    
  summarize(draws);
}
				 

template <Sampler U, typename G>
void test_nuts(const VectorS& theta_init, G& generator, int D, int N,
               S step_size, S max_depth, S max_error, const VectorS& inv_mass) {
  total_time = 0.0;
  count = 0;
  MatrixS draws(D, N);
  std::cout << std::endl
            << "D = " << D << ";  N = " << N << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth
            << ";  WALNUTS = " << (U == Sampler::Walnuts ? "true" : "false")
            << std::endl;

  auto global_start = std::chrono::high_resolution_clock::now();
  if constexpr (U == Sampler::Walnuts) {
    nuts::walnuts(generator, standard_normal_logp_grad, inv_mass, step_size,
                  max_depth, max_error, theta_init, draws);
  } else if constexpr (U == Sampler::Nuts) {
    nuts::nuts(generator, standard_normal_logp_grad, inv_mass, step_size,
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
  summarize(draws);
}

int main() {
  int seed = 428763;
  int D = 200;
  int N = 5000;
  S step_size = 0.25;
  int max_depth = 10;
  S log_max_error = 0.2;  // 80% Metropolis, 45% Barker
  VectorS inv_mass = VectorS::Ones(D);

  std::mt19937 rng(seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  test_nuts<Sampler::Nuts>(theta_init, rng, D, N, step_size, max_depth,
                           log_max_error, inv_mass);
  test_nuts<Sampler::Walnuts>(theta_init, rng, D, N, step_size, max_depth,
                              log_max_error, inv_mass);

  test_walnuts_iter(theta_init, rng, D, N, step_size, max_depth, log_max_error,
		    inv_mass);

  test_adaptive_walnuts(theta_init, rng, D, N, max_depth, log_max_error);

  return 0;
}
