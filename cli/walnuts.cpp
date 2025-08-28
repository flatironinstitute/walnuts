#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <cxxopts.hpp>

#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include <walnuts/util.hpp>
#include "../examples/load_stan.hpp"

static void write_csv(const std::vector<std::string>& names,
		      const Eigen::MatrixXd& draws,
		      const std::string& filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file " + filename);
  }
  out << std::setprecision(12);
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << ",";
    out << names[i];
  }
  out << '\n';
  // TODO: replace with initial call to .transpose()
  for (int col = 0; col < draws.cols(); ++col) {
    for (int row = 0; row < draws.rows(); ++row) {
      if (row > 0) out << ",";
      out << draws(row, col);
    }
    out << '\n';
  }
}

template <typename RNG>
static Eigen::VectorXd std_normal(int N, RNG& rng) {
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd y(N);
  for (long n = 0; n < N; ++n) {
    y(n) = norm(rng);
  }
  return y;
}



template <typename T>
int error_out(const std::string& msg, const T& opts) {
  std::cerr << "ERROR:" << msg << "\n\n" << opts.help() << "\n";
  return 1;
}

template <typename T>
void echo(const std::string& key, const T& val) {
  std::cout << "    --" << key << " " << val << "\n";
}
  
int main(int argc, char** argv) {
  try {
    cxxopts::Options opts("walnuts_cli",
                          "WALNUTS command-line interface");

    std::string stan;
    std::string data;
    std::string out = "out.csv";
    int n_warmup = 128;
    int n_sample = 128;
    unsigned int seed = 42;
    double mass_count = 1.1;
    double mass_offset = 1.1;
    double mass_smooth = 1e-5;
    double step_init = 1.0;
    double accept_target = 0.8;
    double step_offset = 5.0;
    double learning_rate = 1.5;
    double decay_rate = 0.05;
    double max_error = 0.5;
    int max_nuts_depth = 10;
    int max_step_depth = 6;

    opts.positional_help("<stan_file> <data_file>");
    opts.add_options()
      ("h,help", "Show help")
      ("stan", "Compiled Stan .so file",
       cxxopts::value<std::string>(stan), "<path>")
      ("data", "Path to data .json file",
       cxxopts::value<std::string>(data)->default_value(""), "<path>")
      ("out", "Output sample .csv file",
       cxxopts::value<std::string>(out)->default_value("out.csv"), "<path>")
      ("n_warmup", "Warmup iterations",
       cxxopts::value<int>(n_warmup)->default_value("128"), "<int>")
      ("n_sample", "Sampling iterations",
       cxxopts::value<int>(n_sample)->default_value("128"), "<int>")
      ("seed", "Random number generator seed",
       cxxopts::value<int>(n_sample)->default_value("42"), "<uint>")
      ("mass_count", "Pseudocount of initial mass matrix",
       cxxopts::value<double>(mass_count)->default_value("1.1"), "<float>")
      ("mass_offset", "Pseudoposition after initial mass matrix",
       cxxopts::value<double>(mass_offset)->default_value("1.1"), "<float>")
      ("mass_smooth", "Additive smoothing to mass matrix",
       cxxopts::value<double>(mass_smooth)->default_value("1e-5"), "<double>")
      ("step_init", "Initial step size",
       cxxopts::value<double>(step_init)->default_value("1.0"), "<double>")
      ;

    // Treat first two positionals as stan_file and data_file
    // opts.parse_positional({"stan_file", "data_file"});

    auto result = opts.parse(argc, argv);
    if (result.count("help") > 0) {
      std::cout << opts.help() << "\n";
      return 0;
    }

    std::cout << "Parsed arguments:\n";
    echo("stan", stan);
    echo("data", data);
    echo("out", out);
    echo("n_warmup", n_warmup);
    echo("n_sample", n_sample);
    echo("seed", seed);
    echo("mass_count", mass_count);
    echo("mass_offset", mass_offset);
    echo("mass_smooth", mass_smooth);
    echo("step_init", step_init);
    echo("accept_target", accept_target);
    echo("step_offset", step_offset);
    echo("learning_rate", learning_rate);
    echo("decay_rate", decay_rate);
    echo("max_error", max_error);
    echo("max_nuts_depth", max_nuts_depth);
    echo("max_step_depth", max_step_depth);

    if (stan.empty()) {
      return error_out("--stan argument required", opts);
    }
    if (n_warmup < 0) {
      return error_out("n_warmup must be >= 0", opts);
    }
    if (n_sample < 0) {
      return error_out("n_sample must be >= 0", opts);
    }      
    if (!(mass_count >= 1)) {
      return error_out("mass_count must be >= 1", opts);
    }
    if (!(mass_offset >= 1)) {
      return error_out("mass_offset must be >= 1", opts);
    }
    if (!(mass_smooth > 0)) {
      return error_out("mass_smooth must be > 0", opts);
    }
    if (!(step_init > 0)) {
      return error_out("step_init must be > 0", opts);
    }
    if (!(accept_target > 0 && accept_target < 1)) {
      return error_out("accept_target must be > 0 and < 1)", opts);
    }
    if (!(step_offset >= 1)) {           
      return error_out("step_offset must be > 1", opts);
    }                    
    if (!(learning_rate > 0)) {          
      return error_out("learning_rate must be > 0", opts);
    }                    
    if (!(decay_rate > 0)) {
      return error_out("decay_rate must be > 0", opts);
    }                    
    if (!(max_error > 0)) {              
      return error_out("max_error must be > 0", opts);
    }                    
    if (max_nuts_depth <= 0) {
      return error_out("max_nuts_depth must be > 0", opts); 
    }                    
    if (max_step_depth <= 0) {
      return error_out("max_step_depth must be > 0", opts); 
    }
    DynamicStanModel model(stan.c_str(), data.c_str(), seed);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    long logp_grad_calls = 0;
    auto logp = [&](auto&&... args) {
      ++logp_grad_calls;
      model.logp_grad(args...);
    };
      int D = model.unconstrained_dimensions();
      Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(D);
      nuts::MassAdaptConfig mass_cfg(mass_init, mass_count, mass_offset,
				     mass_smooth);
          
      nuts::StepAdaptConfig step_cfg(step_init, accept_target, step_offset,
				     learning_rate, decay_rate);
      nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth);
      Eigen::VectorXd theta_init = std_normal(D, rng);
      nuts::AdaptiveWalnuts walnuts(rng, logp, theta_init,
				    mass_cfg, step_cfg, walnuts_cfg);
      for (long n = 0; n < n_warmup; ++n) {
	walnuts();
      }
      auto sampler = walnuts.sampler();
      int M = model.constrained_dimensions();
      Eigen::MatrixXd draws(M, n_sample);
      for (long n = 0; n < n_sample; ++n) {
	auto draw = sampler();
	model.constrain_draw(draw, draws.col(n));
      }
      write_csv(model.param_names(), draws, out);
                                        
    std::quick_exit(0);      
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Option parse error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
