#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>

#include "load_stan.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Vector = Eigen::Matrix<double, -1, 1>;
using Matrix = Eigen::Matrix<double, -1, -1>;

static void summarize(const std::vector<std::string> names,
                      const Matrix& draws) {
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
    std::cout << names[static_cast<std::size_t>(d)] << ": mean = " << mean
              << ", stddev = " << stddev << "\n";
  }
}

static void write_draws(const std::string& filename,
                        const std::vector<std::string>& names,
                        const Matrix& draws) {
  if (filename.empty()) {
    return;
  }
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << names[i];
  }
  out << "\n";

  auto EigenCommaFormat =
      Eigen::IOFormat(12, Eigen::DontAlignCols, ",", "\n", "", "", "", "");

  out << draws.transpose().format(EigenCommaFormat);
  out.close();
}

template <typename RNG>
Matrix run_walnuts(DynamicStanModel& model, RNG& rng, const Vector& theta_init,
                   std::size_t warmup, std::size_t samples, double init_count,
                   double mass_iteration_offset, double additive_smoothing,
                   double step_size_init, double accept_rate_target,
                   double learn_rate, double beta1, double beta2,
                   double epsilon, double max_error, std::size_t max_nuts_depth,
                   std::size_t max_step_depth, std::size_t min_micro_steps) {
  double logp_time = 0.0;
  std::size_t logp_count = 0;
  auto global_start = std::chrono::high_resolution_clock::now();

  auto end_timing = [&]() {
    auto global_end = std::chrono::high_resolution_clock::now();
    auto global_total_time =
        std::chrono::duration<double>(global_end - global_start).count();
    std::cout << "    total time: " << global_total_time << "s" << std::endl;
    std::cout << "logp_grad time: " << logp_time << "s" << std::endl;
    std::cout << "logp_grad fraction: " << logp_time / global_total_time
              << std::endl;
    std::cout << "        logp_grad calls: " << logp_count << std::endl;
    std::cout << "        time per call: " << logp_time / logp_count << "s"
              << std::endl;
    std::cout << std::endl;
  };

  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(theta_init.size());

  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing);

  nuts::AdamConfig step_cfg(step_size_init, accept_rate_target, learn_rate,
                            beta1, beta2, epsilon);
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth,
                                  min_micro_steps);

  std::cout << "Running Adaptive WALNUTS"
            << ";  D = " << theta_init.size() << "; W = " << warmup
            << ";  N = " << samples << "; step_size_init = " << step_size_init
            << "; max_nuts_depth = " << max_nuts_depth
            << "; max_error = " << max_error << std::endl;

  auto logp = [&](auto&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    model.logp_grad(args...);

    auto end = std::chrono::high_resolution_clock::now();
    logp_time += std::chrono::duration<double>(end - start).count();
    ++logp_count;
  };
  nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init, mass_cfg, step_cfg,
                                walnuts_cfg);

  for (std::size_t w = 0; w < warmup; ++w) {
    walnuts();
  }

  end_timing();

  // N post-warmup draws
  auto sampler = walnuts.sampler();  // freeze tuning
  std::cout << "Adaptation completed." << std::endl;
  std::cout << "Macro step size = " << sampler.macro_step_size() << std::endl;
  std::cout << "Mass matrix diagonal = ["
            << sampler.inverse_mass_matrix_diagonal() << "]" << std::endl;

  Matrix draws(model.constrained_dimensions(), samples);

  logp_time = 0.0;
  logp_count = 0;
  global_start = std::chrono::high_resolution_clock::now();

  for (std::size_t n = 0; n < samples; ++n) {
    model.constrain_draw(sampler(), draws.col(static_cast<Eigen::Index>(n)));
  }

  end_timing();

  return draws;
}

template <typename RNG>
Vector initialize(DynamicStanModel& model, RNG& rng, double init_range,
                  std::size_t max_tries = 100) {
  std::size_t D = model.unconstrained_dimensions();
  std::uniform_real_distribution<double> initial(-init_range, init_range);
  Vector theta_init(D);

  Vector grad(D);
  double logp = 0.0;

  for (std::size_t _ = 0; _ < max_tries; ++_) {
    for (std::size_t i = 0; i < D; ++i) {
      theta_init(static_cast<Eigen::Index>(i)) = initial(rng);
    }

    model.logp_grad(theta_init, logp, grad);
    if (std::isfinite(logp) && grad.allFinite()) {
      // if the log density and gradient are finite, we can use this
      // as the initial point
      std::cout << "Initialized at [" << theta_init.transpose() << "]"
                << std::endl;
      return theta_init;
    }
  }

  throw std::runtime_error("Failed to initialize the model after " +
                           std::to_string(max_tries) + " tries.");
}

int main(int argc, char** argv) {
  unsigned int time_seed = static_cast<unsigned int>(
      std::chrono::system_clock::now().time_since_epoch().count());
  srand(time_seed);
  unsigned int seed = static_cast<unsigned int>(rand());
  std::size_t warmup = 128;
  std::size_t samples = 128;
  std::size_t max_nuts_depth = 10;
  std::size_t max_step_depth = 8;
  std::size_t min_micro_steps = 1;
  double max_error = 0.5;
  double init = 2.0;
  double mass_init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double mass_additive_smoothing = 1e-5;
  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_learn_rate = 0.2;
  double step_beta1 = 0.3;
  double step_beta2 = 0.99;
  double step_epsilon = 1e-4;

  std::string lib;
  std::string data;
  std::string output_file;

  // parse from command line with CLI11
  {
    CLI::App app{"Run WALNUTs on a Stan model"};

    app.add_option("--seed", seed, "Random seed (default randomize with clock)")
        ->default_val(seed);

    app.add_option("--warmup", warmup, "Number of warmup iterations")
        ->default_val(warmup)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--samples", samples, "Number of samples to draw")
        ->default_val(samples)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-depth", max_nuts_depth,
                   "Maximum depth for NUTS trajectory doublings")
        ->default_val(max_nuts_depth)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-step-depth", max_step_depth,
                   "Maximum depth for the step size adaptation")
        ->default_val(max_step_depth)
        ->check(CLI::PositiveNumber);

    app.add_option("--min-micro-steps", min_micro_steps,
                   "Minimum micro steps per macro step")
        ->default_val(min_micro_steps)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-error", max_error,
                   "Maximum error allowed in joint densities")
        ->default_val(max_error)
        ->check(CLI::PositiveNumber);

    app.add_option("--init", init,
                   "Range [-init,init] for uniform parameter initial values")
        ->default_val(init)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--mass-init-count", mass_init_count,
                   "Initial count for the mass matrix adaptation")
        ->default_val(mass_init_count)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-iteration-offset", mass_iteration_offset,
                   "Offset for the mass matrix adaptation iterations")
        ->default_val(mass_iteration_offset)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-additive-smoothing", mass_additive_smoothing,
                   "Additive smoothing for the mass matrix adaptation")
        ->default_val(mass_additive_smoothing)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-size-init", step_size_init,
                   "Initial step size for the step size adaptation")
        ->default_val(step_size_init)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-accept-rate-target", accept_rate_target,
                   "Target acceptance rate for the step size adaptation")
        ->default_val(accept_rate_target)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-learning-rate", step_learn_rate,
                   "Learning rates for step adaptation")
        ->default_val(step_learn_rate)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-beta1", step_beta1,
                   "Decay rate of gradient moving average for step adaptation")
        ->default_val(step_beta1)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option(
           "--step-beta2", step_beta1,
           "Decay rate of squared gradient moving average for step adaptation")
        ->default_val(step_beta2)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-epsilon", step_epsilon,
                   "Update stabilization term for step size adaptation")
        ->default_val(step_epsilon)
        ->check(CLI::PositiveNumber);

    app.add_option("model", lib,
                   "Path to the Stan model library (.so from CmdStan{,Py,R})")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("data", data,
                   "Path to the Stan model data (.json, optional)")
        ->check(CLI::ExistingFile);

    app.add_option("--output", output_file, "Output file for the draws")
        ->check(CLI::NonexistentPath);

    CLI11_PARSE(app, argc, argv);
  }

  DynamicStanModel model(lib.c_str(), data.c_str(), seed);

  std::mt19937 rng(seed);

  Vector theta_init = initialize(model, rng, init);

  Matrix draws = run_walnuts(
      model, rng, theta_init, warmup, samples, mass_init_count,
      mass_iteration_offset, mass_additive_smoothing, step_size_init,
      accept_rate_target, step_learn_rate, step_beta1, step_beta2, step_epsilon,
      max_error, max_nuts_depth, max_step_depth, min_micro_steps);

  auto names = model.param_names();
  summarize(names, draws);
  write_draws(output_file, names, draws);

  return 0;
}
