
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "walnuts/api.hpp"
#include "walnuts/config.hpp"
#include "walnuts/validate.hpp"

struct GlobalHandler {
  void on_r_hat(double r_hat) {
    r_hats_.push_back(r_hat);
  }

  std::vector<double> r_hats_;
};

struct CacheHandler {
  CacheHandler(bool save_warmup = true) : save_warmup_(save_warmup) {}

  void on_warmup(const Eigen::VectorXd& position, const double lp,
                 const double stepsize, const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    warmup_draws_.push_back(position);
    warmup_lps_.push_back(lp);
    warmup_stepsizes_.push_back(stepsize);
    warmup_diag_inv_masses_.push_back(diag_inv_mass);
  }

  void on_warmup_complete(double stepsize,
                          const Eigen::VectorXd& diag_inv_mass) {
    stepsize_ = stepsize;
    diag_inv_mass_ = diag_inv_mass;
  }

  void on_sample(const Eigen::VectorXd& position, double lp) {
    draws_.push_back(position);
    lps_.push_back(lp);
  }

  void on_stop() {
    // TODO: add stop semaphore, catch interrupts
  }

  static void write_step_size_csv(const std::vector<CacheHandler>& handlers,
                                  std::ostream& os, int precision = 15) {
    if (handlers.empty()) {
      return;
    }
    os << std::setprecision(precision);
    os << "chain_id,stepsize\n";
    for (std::size_t c = 0; c < handlers.size(); ++c) {
      os << c << ',' << handlers[c].stepsize_ << '\n';
    }
  }

  static void write_mass_matrix_csv(const std::vector<CacheHandler>& handlers,
                                    std::ostream& os, int precision = 15) {
    if (handlers.empty()) {
      return;
    }
    std::int64_t D = handlers[0].diag_inv_mass_.size();
    if (D == 0) {
      return;
    }
    os << std::setprecision(precision);
    os << "chain_id";
    for (std::int64_t d = 0; d < D; ++d) {
      os << ",theta[" << d << "]";
    }
    os << '\n';
    for (std::size_t c = 0; c < handlers.size(); ++c) {
      os << c;
      for (int d = 0; d < D; ++d) {
        os << ',' << handlers[c].diag_inv_mass_(d);
      }
      os << '\n';
    }
  }

  static void write_sample_csv(const std::vector<CacheHandler>& handlers,
                               std::ostream& os, bool include_warmup = true,
                               int precision = 8) {
    os << std::setprecision(precision);

    // get dims D from first draw
    int64_t D = 0;
    for (const auto& h : handlers) {
      if (!h.draws_.empty()) {
        D = h.draws_[0].size();
        break;
      }
      if (!h.warmup_draws_.empty()) {
        D = h.warmup_draws_[0].size();
        break;
      }
    }
    if (D == 0) {
      return;
    }

    os << "chain_id,warmup,iteration,log_density";
    for (int d = 0; d < D; ++d) {
      os << ",theta[" << d << "]";
    }
    os << '\n';

    for (std::size_t c = 0; c < handlers.size(); ++c) {
      const auto& h = handlers[c];
      int iteration = 0;

      if (include_warmup) {
        for (std::size_t n = 0; n < h.warmup_draws_.size(); ++n, ++iteration) {
          os << c << ",1," << iteration << ',' << h.warmup_lps_[n];
          for (int d = 0; d < D; ++d) {
            os << ',' << h.warmup_draws_[n](d);
          }
          os << '\n';
        }
      }

      for (std::size_t n = 0; n < h.draws_.size(); ++n, ++iteration) {
        os << c << ",0," << iteration << ',' << h.lps_[n];
        for (int d = 0; d < D; ++d) {
          os << ',' << h.draws_[n](d);
        }
        os << '\n';
      }
    }
  }

  static void write_step_size_csv(const std::vector<CacheHandler>& handlers,
                                  const std::string& filename,
                                  int precision = 8) {
    std::ofstream os(filename);
    walnuts::validate_open(os, filename);
    write_step_size_csv(handlers, os, precision);
  }

  static void write_mass_matrix_csv(const std::vector<CacheHandler>& handlers,
                                    const std::string& filename,
                                    int precision = 8) {
    std::ofstream os(filename);
    walnuts::validate_open(os, filename);
    write_mass_matrix_csv(handlers, os, precision);
  }

  static void write_sample_csv(const std::vector<CacheHandler>& handlers,
                               const std::string& filename,
                               bool include_warmup = true, int precision = 8) {
    std::ofstream os(filename);
    walnuts::validate_open(os, filename);
    write_sample_csv(handlers, os, include_warmup, precision);
  }

  bool save_warmup_;

  double stepsize_ = 0;
  Eigen::VectorXd diag_inv_mass_;

  std::vector<double> r_hats_;
  
  std::vector<Eigen::VectorXd> draws_;
  std::vector<double> lps_;

  std::vector<Eigen::VectorXd> warmup_draws_;
  std::vector<double> warmup_lps_;
  std::vector<double> warmup_stepsizes_;
  std::vector<Eigen::VectorXd> warmup_diag_inv_masses_;
};

static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

int main() {
  auto logp_grad = std_normal;

  uint32_t seed = 48;
  std::seed_seq seed_seq_for_init{seed, 0u};
  std::mt19937 rng{seed_seq_for_init};
  uint64_t num_chains = 32;
  uint64_t dims = 100;

  std::vector<CacheHandler> handlers(num_chains);
  for (size_t m = 0; m < num_chains; ++m) {
    handlers[m] = CacheHandler();
  }
  GlobalHandler global_handler;

  double init_scale = 0.5;
  double mass_smoothing = 0.1;
  auto init_cfg = walnuts::InitConfigBuilder(num_chains, dims)
                      .positions(rng, init_scale)
                      .masses(std_normal, mass_smoothing)
                      .build();

  auto warmup_cfg = walnuts::WarmupConfigBuilder()
                        .min_max_iter(500, 2000)
                        .step_size_converge_tol(1)
                        .mass_init_count(4.0)
                        .build();

  auto sampling_cfg = walnuts::SamplingConfigBuilder()
                          .min_max_iter(100, 1000)
                          .max_trajectory_doublings(8)
                          .rhat_converge_tol(1.0001)
                          .build();

  // std::cout << init_cfg << "\n\n";  // too verbose with multi-chain
  std::cout << warmup_cfg << "\n\n";
  std::cout << sampling_cfg << "\n\n";

  walnuts::walnuts(seed, handlers, global_handler, logp_grad, init_cfg, warmup_cfg,
                   sampling_cfg);

  std::cout << "PER-CHAIN STATISTICS: " << "\n";
  for (size_t m = 0; m < num_chains; ++m) {
    std::cout << "  Chain " << m
	      << "; step size = " << handlers[m].stepsize_
	      << "; ||mass|| = " << handlers[m].diag_inv_mass_.array().inverse().matrix().norm()
              << "; # warmup_draws = " << handlers[m].warmup_draws_.size()
              << "; # draws = " << handlers[m].draws_.size() << "\n";
  }
  std::cout << "\n";

  std::cout << "NUMBER OF R-HAT EVALS: " << global_handler.r_hats_.size()
	    << ";  FINAL R-HAT: " << global_handler.r_hats_.back() << "\n\n";

  std::cout << "WRITING TO FILES: step_size.csv, mass_matrix.csv, sample.csv\n\n";

  CacheHandler::write_step_size_csv(handlers, "step_size.csv");
  CacheHandler::write_mass_matrix_csv(handlers, "mass_matrix.csv");
  CacheHandler::write_sample_csv(handlers, "sample.csv");

  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}
