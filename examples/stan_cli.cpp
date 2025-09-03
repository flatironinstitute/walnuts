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
    std::cout << names[d] << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
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
                   int64_t warmup, int64_t samples, double init_count,
                   double mass_iteration_offset, double additive_smoothing,
                   double step_size_init, double accept_rate_target,
                   double step_iteration_offset, double learning_rate,
                   double decay_rate, double max_error, int64_t max_nuts_depth,
                   int64_t max_step_depth) {
  double logp_time = 0.0;
  int logp_count = 0;
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

  nuts::StepAdaptConfig step_cfg(step_size_init, accept_rate_target,
                                 step_iteration_offset, learning_rate,
                                 decay_rate);
  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);

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

  for (auto w = 0; w < warmup; ++w) {
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

  for (auto n = 0; n < samples; ++n) {
    model.constrain_draw(sampler(), draws.col(n));
  }

  end_timing();

  return draws;
}

template <typename RNG>
Vector initialize(DynamicStanModel& model, RNG& rng, double init_range,
                  int64_t max_tries = 100) {
  int D = model.unconstrained_dimensions();
  std::uniform_real_distribution<double> initial(-init_range, init_range);
  Vector theta_init(D);

  Vector grad(D);
  double logp = 0.0;

  for (auto _ = 0; _ < max_tries; ++_) {
    for (auto i = 0; i < D; ++i) {
      theta_init(i) = initial(rng);
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
  srand(std::chrono::system_clock::now().time_since_epoch().count());
  unsigned int seed = rand();
  int64_t warmup = 128;
  int64_t samples = 128;
  int64_t max_nuts_depth = 10;
  int64_t max_step_depth = 8;
  double max_error = 0.5;
  double init = 2.0;
  double init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double additive_smoothing = 1e-5;
  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_iteration_offset = 5.0;
  double learning_rate = 1.5;
  double decay_rate = 0.05;

  std::string lib;
  std::string data;
  std::string output_file;

  // parse from command line with CLI11
  {
    CLI::App app{"Run WALNUTs on a Stan model"};

    app.add_option("--seed", seed, "Random seed")->default_val(seed);

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

    app.add_option("--max-error", max_error,
                   "Maximum error allowed in joint densities")
        ->default_val(max_error)
        ->check(CLI::PositiveNumber);

    app.add_option("--init", init,
                   "Range [-init,init] for the parameters initial values")
        ->default_val(init)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--mass-init-count", init_count,
                   "Initial count for the mass matrix adaptation")
        ->default_val(init_count)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-iteration-offset", mass_iteration_offset,
                   "Offset for the mass matrix adaptation iterations")
        ->default_val(mass_iteration_offset)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-additive-smoothing", additive_smoothing,
                   "Additive smoothing for the mass matrix adaptation")
        ->default_val(additive_smoothing)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-size-init", step_size_init,
                   "Initial step size for the step size adaptation")
        ->default_val(step_size_init)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-accept-rate-target", accept_rate_target,
                   "Target acceptance rate for the step size adaptation")
        ->default_val(accept_rate_target)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-iteration-offset", step_iteration_offset,
                   "Offset for the step size adaptation iterations")
        ->default_val(step_iteration_offset)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--step-learning-rate", learning_rate,
                   "Learning rate for the step size adaptation")
        ->default_val(learning_rate)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-decay-rate", decay_rate,
                   "Decay rate for the step size adaptation")
        ->default_val(decay_rate)
        ->check(CLI::PositiveNumber);

    app.add_option("model", lib, "Path to the Stan model library")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("data", data, "Path to the Stan model data (optional)")
        ->check(CLI::ExistingFile);

    app.add_option("--output", output_file, "Output file for the draws")
        ->check(CLI::NonexistentPath);

    CLI11_PARSE(app, argc, argv);
  }

  DynamicStanModel model(lib.c_str(), data.c_str(), seed);

  std::mt19937 rng(seed);

  Vector theta_init = initialize(model, rng, init);

  Matrix draws =
      run_walnuts(model, rng, theta_init, warmup, samples, init_count,
                  mass_iteration_offset, additive_smoothing, step_size_init,
                  accept_rate_target, step_iteration_offset, learning_rate,
                  decay_rate, max_error, max_nuts_depth, max_step_depth);

  auto names = model.param_names();
  summarize(names, draws);
  write_draws(output_file, names, draws);

  return 0;
}
