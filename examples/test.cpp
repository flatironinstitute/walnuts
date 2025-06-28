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
using Integer = long;

static double total_time = 0.0;
static Integer count = 0;
static auto global_start = std::chrono::high_resolution_clock::now();
static auto block_start = std::chrono::high_resolution_clock::now();

static void global_start_timer() {
  count = 0;
  global_start = std::chrono::high_resolution_clock::now();
  total_time = 0;
}

static void global_end_timer() {
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time =
      std::chrono::duration<double>(global_end - global_start).count();
  std::cout << "     logp_grad calls: " << count << std::endl;
  std::cout << "          total time: " << global_total_time
	    << "s" << std::endl;
  std::cout << "      logp_grad time: " << total_time << "s" << std::endl;
  std::cout << "  logp_grad fraction: " << total_time / global_total_time
            << std::endl;
  std::cout << std::endl;
}

static void block_start_timer() {
  block_start = std::chrono::high_resolution_clock::now();
}

static void block_end_timer() {
  auto end = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end - block_start).count();
  ++count;
}

static void ill_cond_normal_logp_grad(const VectorS& x,
				      S& logp,
				      VectorS& grad) {
  block_start_timer();
  Integer D = x.size();
  grad = VectorS::Zero(D);
  logp = 0;
  for (Integer d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
  block_end_timer();
}

static void std_normal_logp_grad(const VectorS& x,
				 S& logp,
				 VectorS& grad) {
  block_start_timer();
  logp = -0.5 * x.dot(x);
  grad = -x;
  block_end_timer();
}


static void summarize(const MatrixS& draws) {
  Integer N = draws.cols();
  Integer D = draws.rows();
  for (Integer d = 0; d < D; ++d) {
    if (d > 3 && d < D - 3) {
      if (d == 4) {
	std::cout << "... elided " << (D - 6) << " rows ..." << std::endl;
      }
      continue;
    }
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
}


template <typename F, typename RNG>
static void test_nuts(const F& target_logp_grad, const VectorS& theta_init, RNG& rng, Integer D, Integer N,
		      S step_size, Integer max_depth, const VectorS& inv_mass) {
  std::cout << "\nTEST NUTS"
	    << ";  D = " << D
	    << ";  N = " << N
	    << ";  step_size = " << step_size
	    << ";  max_depth = " << max_depth
	    << std::endl;
  global_start_timer();
  nuts::Random<double, RNG> rand(rng);
  nuts::Nuts sample(rand, target_logp_grad, theta_init, inv_mass, step_size, max_depth);
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    draws.col(n) = sample();
  }
  global_end_timer();
  summarize(draws);
}


template <typename F, typename RNG>
static void test_walnuts(const F& target_logp_grad, VectorS theta_init, RNG& rng, Integer D, Integer N,
			 S macro_step_size, Integer max_nuts_depth, S max_error,
			 VectorS inv_mass) {
  std::cout << "\nTEST WALNUTS"
	    << ";  D = " << D
	    << ";  N = " << N
	    << ";  macro_step_size = " << macro_step_size
	    << ";  max_nuts_depth = " << max_nuts_depth
	    << ";  max_error = " << max_error
	    << std::endl;
  global_start_timer();
  nuts::Random<double, RNG> rand(rng);
  nuts::WalnutsSampler sample(rand, target_logp_grad, theta_init,
			      inv_mass, macro_step_size, max_nuts_depth,
			      max_error);
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    draws.col(n) = sample();
  }
  global_end_timer();
  summarize(draws);
}


template <typename F, typename RNG>
static void test_adaptive_walnuts(const F& target_logp_grad,
				  const VectorS& theta_init, RNG& rng, Integer D, Integer N,
				  Integer max_nuts_depth, S max_error) {
  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
  double init_count = 10.0;
  double mass_iteration_offset = 4.0;
  double additive_smoothing = 0.05;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
				 additive_smoothing);
  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_iteration_offset = 4.0;
  double learning_rate = 0.95;
  double decay_rate = 0.05;
  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
				 step_iteration_offset, learning_rate,
				 decay_rate);
  Integer max_step_depth = 8;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth,
				  max_step_depth);
  std::cout << "\nTEST ADAPTIVE WALNUTS"
	    << ";  D = " << D
	    << ";  N = " << N
	    << "; step_size_init = " << step_size_init
	    << "; max_nuts_depth = " << max_nuts_depth
	    << "; max_error = " << max_error
	    << std::endl;
  global_start_timer();
  nuts::AdaptiveWalnuts walnuts(rng, target_logp_grad, theta_init, mass_cfg,
				step_cfg, walnuts_cfg);
  for (Integer n = 0; n < N; ++n) {
    walnuts();
  }
  auto sampler = walnuts.sampler();
  MatrixS draws(D, N);
  for (Integer n = 0; n < N; ++n) {
    draws.col(n) = sampler();
  }
  global_end_timer();
  summarize(draws);
  std::cout << std::endl;
  std::cout << "Macro step size = " << sampler.macro_step_size() << std::endl;
  std::cout << "Max error = " << sampler.max_error() << std::endl;
  std::cout << "Inverse mass matrix = "
	    << sampler.inverse_mass_matrix_diagonal().transpose() << std::endl;
}

int main() {
  unsigned int seed = 428763;
  Integer D = 200;
  Integer N = 1000;
  S step_size = 0.465;
  Integer max_depth = 10;
  S max_error = 1.0;  // 61% Metropolis
  VectorS inv_mass = VectorS::Ones(D);
  std::mt19937 rng(seed);

  // std::normal_distribution<S> std_normal(0, 1);
  // VectorS theta_init(D);
  // for (Integer i = 0; i < D; ++i) {
  //   theta_init(i) = std_normal(rng);
  // }
  VectorS theta_init = VectorS::Zero(D);

  // auto target_logp_grad = std_normal_logp_grad;
  auto target_logp_grad = ill_cond_normal_logp_grad;

  test_nuts(target_logp_grad, theta_init, rng, D, N, step_size, max_depth, inv_mass);

  test_walnuts(target_logp_grad, theta_init, rng, D, N, step_size, max_depth, max_error,
	       inv_mass);

  test_adaptive_walnuts(target_logp_grad, theta_init, rng, D, N, max_depth, max_error);

  return 0;
}
