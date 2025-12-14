#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>

static double total_time = 0.0;
static std::size_t count = 0;
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
  std::cout << "          total time: " << global_total_time << "s"
            << std::endl;
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

// p(y) = normal(y | 0, I)
static void std_normal_logp_grad(const Eigen::VectorXd& x, double& logp,
                                 Eigen::VectorXd& grad) {
  block_start_timer();
  logp = -0.5 * x.dot(x);
  grad = -x;
  block_end_timer();
}

// p(y) = normal(0, diag(sigma)), sigma[d] = d
static void ill_cond_normal_logp_grad(const Eigen::VectorXd& x, double& logp,
                                      Eigen::VectorXd& grad) {
  block_start_timer();
  const auto D = x.size();
  grad = Eigen::VectorXd::Zero(D);
  logp = 0;
  for (auto d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
  block_end_timer();
}

// p(y) = normal(y | 0, Sigma), with Sigma[i, j] = rho^abs(i - j)
static void rw1_logp_grad(const Eigen::VectorXd& y,
			  double& logp, Eigen::VectorXd& grad) {
  block_start_timer();
  double rho = 0.99;
  Eigen::Index D = y.size();
  double sigma_sq = 1.0 - rho * rho;
  double inv_sigma_sq = 1.0 / sigma_sq;
  grad.setZero(D);
  logp = -0.5 * y[0] * y[0];
  grad[0] -= y[0];
  for (Eigen::Index n = 1; n < D; ++n) {
    double r = y[n] - rho * y[n - 1];
    double w = r * inv_sigma_sq;
    logp -= 0.5 * r * w;
    grad[n]   -= w;
    grad[n-1] += rho * w;
  }
  block_end_timer();
}


static void summarize(const Eigen::MatrixXd& draws) {
  auto N = draws.cols();
  auto D = draws.rows();
  for (auto d = 0; d < D; ++d) {
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
static void run_nuts(const F& target_logp_grad,
                     const Eigen::VectorXd& theta_init, RNG& rng, std::size_t D,
                     std::size_t N, double step_size, std::size_t max_depth,
                     const Eigen::VectorXd& inv_mass) {
  std::cout << "\nRUN NUTS"
            << ";  D = " << D << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << std::endl;
  global_start_timer();
  nuts::Random<double, RNG> rand(rng);
  nuts::Nuts sample(rand, target_logp_grad, theta_init, inv_mass, step_size,
                    max_depth);
  Eigen::MatrixXd draws(static_cast<Eigen::Index>(D),
                        static_cast<Eigen::Index>(N));
  for (std::size_t n = 0; n < N; ++n) {
    draws.col(static_cast<Eigen::Index>(n)) = sample();
  }
  global_end_timer();
  summarize(draws);
}

template <typename F, typename RNG>
static void run_walnuts(const F& target_logp_grad, Eigen::VectorXd theta_init,
                        RNG& rng, std::size_t D, std::size_t N,
                        double macro_step_size, std::size_t max_nuts_depth,
                        std::size_t max_step_halvings,
                        std::size_t min_micro_steps, double max_error,
                        Eigen::VectorXd inv_mass) {
  std::cout << "\nRUN WALNUTS"
            << ";  D = " << D << ";  N = " << N
            << ";  macro_step_size = " << macro_step_size
            << ";  max_nuts_depth = " << max_nuts_depth
            << ";  min_micro_steps = " << min_micro_steps
            << ";  max_error = " << max_error << std::endl;
  global_start_timer();
  nuts::Random<double, RNG> rand(rng);
  nuts::WalnutsSampler sample(rand, target_logp_grad, theta_init, inv_mass,
                              macro_step_size, max_nuts_depth,
                              max_step_halvings, min_micro_steps, max_error);
  Eigen::MatrixXd draws(D, N);
  for (std::size_t n = 0; n < N; ++n) {
    draws.col(static_cast<Eigen::Index>(n)) = sample();
  }
  global_end_timer();
  summarize(draws);
}

template <typename F, typename RNG>
static void run_adaptive_walnuts(
    const F& target_logp_grad, const Eigen::VectorXd& theta_init, RNG& rng,
    std::size_t D, std::size_t N, double step_size_init,
    std::size_t max_nuts_depth, std::size_t min_micro_steps, double max_error) {
  Eigen::VectorXd mass_init =
      Eigen::VectorXd::Ones(static_cast<Eigen::Index>(D));
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 0.1;
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);

  double accept_rate_target = 0.8;
  double learn_rate = 0.2;
  double beta1 = 0.3;
  double beta2 = 0.99;
  double epsilon = 1e-4;
  nuts::AdamConfig step_cfg(step_size_init, accept_rate_target, learn_rate,
                            beta1, beta2, epsilon);

  std::size_t max_step_depth = 8;
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth,
                                  min_micro_steps);
  double target_depth = 1.5;  // keeping thislow encourages min micro per macro > 2
  std::cout << "\nRUN ADAPTIVE WALNUTS"
            << ";  D = " << D << ";  N = " << N
            << "; step_size_init = " << step_size_init
            << "; max_nuts_depth = " << max_nuts_depth
            << "; min_micro_steps = " << min_micro_steps
            << "; max_error = " << max_error << std::endl;
  global_start_timer();
  nuts::AdaptiveWalnuts adapt(rng, target_logp_grad, theta_init, mass_cfg,
			      step_cfg, walnuts_cfg, target_depth);
  for (std::size_t n = 0; n < N; ++n) {
    adapt();
  }
  auto sample = adapt.sampler();
  Eigen::MatrixXd draws(static_cast<Eigen::Index>(D),
                        static_cast<Eigen::Index>(N));
  for (std::size_t n = 0; n < N; ++n) {
    draws.col(static_cast<Eigen::Index>(n)) = sample();
  }
  global_end_timer();
  summarize(draws);
  std::cout << std::endl;
  std::cout << "Micro step size = " << adapt.step_size()
            << std::endl;
  std::cout << "Min micro steps per macro step = " << adapt.min_micro_steps()
            << std::endl;
  std::cout << "Inverse mass matrix = "
	    << std::fixed << std::setprecision(2)
	    << adapt.inv_mass().transpose() << std::endl;
}

int main() {
  unsigned int seed = 83435638;
  std::size_t D = 100;
  std::size_t N = 1000;
  double step_size = 0.5;
  std::size_t max_depth = 8;
  std::size_t max_step_halvings = 5;
  std::size_t min_micro_steps = 3;
  double max_error = 1;  // 61% Metropolis
  Eigen::VectorXd inv_mass =
      Eigen::VectorXd::Ones(static_cast<Eigen::Index>(D));
  std::mt19937 rng(seed);

  // use this for random standard normal init
  Eigen::VectorXd theta_init(D);
  std::normal_distribution<double> std_normal(0, 1);
  for (std::size_t i = 0; i < D; ++i) {
    theta_init(static_cast<Eigen::Index>(i)) = std_normal(rng);
  }
  // or the following to init at bottleneck
  // theta_init = ::Zero(D);

  // auto target_logp_grad = std_normal_logp_grad;
  // auto target_logp_grad = ill_cond_normal_logp_grad;
  auto target_logp_grad = rw1_logp_grad;
    
  run_nuts(target_logp_grad, theta_init, rng, D, N, step_size, max_depth,
           inv_mass);

  run_walnuts(target_logp_grad, theta_init, rng, D, N, step_size, max_depth,
              max_step_halvings, min_micro_steps, max_error, inv_mass);

  run_adaptive_walnuts(target_logp_grad, theta_init, rng, D, N, step_size,
                       max_depth, min_micro_steps, max_error);

  return 0;
}
