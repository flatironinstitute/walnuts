#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/validate.hpp>
#include <walnuts/util.hpp>
#include <walnuts/config.hpp>

namespace nuts {

  void valiate_probability(double p, const std::string& var) {
    if (!(p > 0 && p < 1))
      throw std::invalid_argument(var + " must be a proportion in (0, 1)");
  }
  
  template <typename T>
  void validate_size(const std::vector<T>& x, uint64_t size,
		     const std::string& var, const std::string& target) {
    if (x.size() != size)
      throw std::invalid_argument(var + " size must match " + target);
  }

  template <typename T, int R, int C>
  void validate_size(const Eigen::Matrix<T, R, C>& x, uint64_t size,
		     const std::string& var, const std::string& target) {
    if (x.size() != static_cast<int64_t>(size))
      throw std::invalid_argument(var + " size must match " + target);
  }

  void validate_finite_gt1(double x, const std::string& var) {
    if (!(std::isfinite(x) && x > 1))
      throw std::invalid_argument(var + " must be finite and > 1");
  }
  
  void validate_finite_positive(uint64_t x, const std::string& var) {
    if (!(std::isfinite(x) && x > 0))
      throw std::invalid_argument(var + " must be > 0");
  }

  void validate_finite_positive(double x, const std::string& var) {
    if (!(std::isfinite(x) && x > 0))
      throw std::invalid_argument(var + " must be finite and > 0");
  }
  
  template <typename T, int R, int C>
  void validate_finite_positive(const Eigen::Matrix<T, R, C>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (int64_t i = 0; i < xs.size(); ++i)
      validate_finite_positive(xs(i), var_entries);
  }

  template <typename T>
  void validate_finite_positive(const std::vector<T>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (const auto& x : xs)
      validate_finite_positive(x, var_entries);
  }

  
  class InitConfig {
  public:
    uint64_t num_chains()                              const { return step_sizes_.size(); }
    const std::vector<double>& step_sizes()            const { return step_sizes_; }
    const std::vector<Eigen::VectorXd>& positions()    const { return positions_; }
    const std::vector<Eigen::VectorXd>& masses()       const { return masses_; }

  private:
    friend class InitConfigBuilder;

    InitConfig() = default;

    std::vector<double> step_sizes_;
    std::vector<Eigen::VectorXd> positions_;
    std::vector<Eigen::VectorXd> masses_;
  };

  class InitConfigBuilder {
  public:
    explicit InitConfigBuilder(uint64_t num_chains, uint64_t dims)
      : num_chains_(num_chains), dims_(dims) {
      this->step_sizes(0.1);
      Eigen::VectorXd position = Eigen::VectorXd::Zero(static_cast<int64_t>(dims));
      this->positions(position);
      Eigen::VectorXd mass = Eigen::VectorXd::Ones(static_cast<int64_t>(dims));
      this->masses(mass);
    }

    InitConfigBuilder& step_sizes(double v) {
      validate_finite_positive(v, "step size");
      step_sizes_ = std::vector<double>(num_chains_, v);
      return *this;
    }
    InitConfigBuilder& step_sizes(const std::vector<double>& v) {
      validate_size(v, num_chains_, "step_sizes", "num_chains");
      validate_finite_positive(v, "step_size");
      step_sizes_ = v;
      return *this;
    }

    template <typename RNG>
    InitConfigBuilder& positions(RNG& rng, double init_scale) {
      Random<double, RNG> rand(rng);
      for (size_t c = 0; c < num_chains_; ++c) {
	positions_[c] = init_scale * rand.standard_normal(dims_);
      }
      return *this;
    }
    InitConfigBuilder& positions(const Eigen::VectorXd& v) {
      validate_size(v, dims_, "position", "dims");
      positions_ = std::vector<Eigen::VectorXd>(num_chains_, v);
      return *this;
    }
    InitConfigBuilder& positions(const std::vector<Eigen::VectorXd>& v) {
      validate_size(v, num_chains_, "positions", "num_chains");
      validate_finite_positive(v, "position");
      positions_ = v;
      return *this;
    }
  
    template <typename LPG>
    InitConfigBuilder& masses(const LPG& lp_grad, double mass_smoothing) {
      validate_finite_positive(mass_smoothing, "mass_smoothing");
      Eigen::VectorXd grad;
      double lp;
      for (size_t c = 0; c < num_chains_; ++c) {
	lp_grad(positions_[c], lp, grad);
	masses_[c] = (1 - mass_smoothing) * grad.array().abs().sqrt() + mass_smoothing;
      }
      return *this;
    }
    InitConfigBuilder& masses(const Eigen::VectorXd& v) {
      validate_size(v, dims_, "mass", "dims");
      validate_finite_positive(v, "masses");
      masses_ = std::vector<Eigen::VectorXd>(num_chains_, v);
      return *this;
    }
    InitConfigBuilder& masses(const std::vector<Eigen::VectorXd>& v) {
      validate_size(v, num_chains_, "masses", "num_chains");
      validate_finite_positive(v, "masses");
      for (const auto& x : v)
	validate_size(x, dims_, "all masses", "dims");
      masses_ = v;
      return *this;
    }

    InitConfig build() {
      InitConfig cfg;
      cfg.step_sizes_ = std::move(step_sizes_);
      cfg.positions_  = std::move(positions_);
      cfg.masses_     = std::move(masses_);
      return cfg;
    }

  private:
    uint64_t num_chains_;
    uint64_t dims_;
    std::vector<double> step_sizes_;
    std::vector<Eigen::VectorXd> positions_;
    std::vector<Eigen::VectorXd> masses_;
  };

  std::ostream& operator<<(std::ostream& out, const InitConfig& cfg) {
    out << "InitConfigs (by chain)\n";
    for (size_t n = 0; n < cfg.step_sizes().size(); ++n) {
      if (n > 0) out << "\n";
      out << "  chain                    = " << n                          << "\n"
	  << "    num_chains             = " << cfg.num_chains()           << "\n"
	  << "    step_size              = " << cfg.step_sizes()[n]        << "\n"
	  << "    position               = " << cfg.positions()[n].transpose() << "\n"
	  << "    mass                   = " << cfg.masses()[n].transpose() << "\n";
    }
    return out;
  }

  class WarmupConfig {
  public: 
    uint64_t max_warmup_iter()          const { return max_warmup_iter_; }
    double step_size_converge_tol()     const { return step_size_converge_tol_; }
    double mass_matrix_converge_tol()   const { return mass_matrix_converge_tol_; }
    double mass_init_count()            const { return mass_init_count_; }
    double mass_additive_smoothing()    const { return mass_additive_smoothing_; }
    double max_macro_steps_target()     const { return max_macro_steps_target_; }
    double step_accept_rate_target()    const { return step_accept_rate_target_; }
    double step_learning_rate()         const { return step_learning_rate_; }
    double step_gradient_decay()        const { return step_gradient_decay_; }
    double step_sq_gradient_decay()     const { return step_sq_gradient_decay_; }
    double step_stabilization()         const { return step_stabilization_; }
  private:
    friend class WarmupConfigBuilder;

    WarmupConfig() = default;

    uint64_t max_warmup_iter_          = 1000;
    double step_size_converge_tol_     = 0.1;
    double mass_matrix_converge_tol_   = 1.0;
    double mass_init_count_            = 4.0;
    double mass_additive_smoothing_    = 1e-5;
    double max_macro_steps_target_     = 15.0;
    double step_accept_rate_target_    = 0.8;
    double step_learning_rate_         = 0.2;
    double step_gradient_decay_        = 0.3;
    double step_sq_gradient_decay_     = 0.99;
    double step_stabilization_         = 1e-4;
  };

  class WarmupConfigBuilder {
  public:
    WarmupConfigBuilder max_warmup_iter(uint64_t v) {
      cfg_.max_warmup_iter_ = v;
      return *this;
    }
    WarmupConfigBuilder step_size_converge_tol(double v) {
      validate_finite_positive(v, "step_size_converge_tol");
      cfg_.step_size_converge_tol_ = v;
      return *this;
    }
    WarmupConfigBuilder mass_matrix_converge_tol(double v) {
      validate_finite_positive(v, "mass_matrix_converge_tol");
      cfg_.mass_matrix_converge_tol_ = v;
      return *this;
    }
    WarmupConfigBuilder mass_init_count(double v) {
      validate_finite_positive(v, "mass_init_count");
      cfg_.mass_init_count_ = v;
      return *this;
    }
    WarmupConfigBuilder mass_additive_smoothing(double v) {
      validate_finite_positive(v, "mass_additive_smoothing");
      cfg_.mass_additive_smoothing_ = v;
      return *this;
    }
    WarmupConfigBuilder max_macro_steps_target(double v) {
      validate_finite_positive(v, "max_macro_steps_target");
      cfg_.max_macro_steps_target_ = v;
      return *this;
    }
    WarmupConfigBuilder step_accept_rate_target(double v) {
      validate_probability(v, "step_accept_rate_target");
      cfg_.step_accept_rate_target_ = v;
      return *this;
    }
    WarmupConfigBuilder step_learning_rate(double v) {
      validate_finite_positive(v, "step_learning_rate");
      cfg_.step_learning_rate_ = v;
      return *this;
    }
    WarmupConfigBuilder step_gradient_decay(double v) {
      validate_probability(v, "step_gradient_decay");
      cfg_.step_gradient_decay_ = v;
      return *this;
    }
    WarmupConfigBuilder step_sq_gradient_decay(double v) {
      validate_probability(v, "step_sq_gradient_decay");
      cfg_.step_sq_gradient_decay_ = v;
      return *this;
    }
    WarmupConfigBuilder step_stabilization(double v) {
      validate_finite_positive(v, "step_stabilization");
      cfg_.step_stabilization_ = v;
      return *this;
    }

    WarmupConfig build() { return cfg_; }

  private:
    WarmupConfig cfg_;
  };

  std::ostream& operator<<(std::ostream& out, const nuts::WarmupConfig& cfg) {
    out << "WarmupConfig\n"
	<< "  max_warmup_iter            = " << cfg.max_warmup_iter()          << "\n"
	<< "  step_size_converge_tol     = " << cfg.step_size_converge_tol()   << "\n"
	<< "  mass_matrix_converge_tol   = " << cfg.mass_matrix_converge_tol() << "\n"
	<< "  mass_init_count            = " << cfg.mass_init_count()          << "\n"
	<< "  mass_additive_smoothing    = " << cfg.mass_additive_smoothing()  << "\n"
	<< "  max_macro_steps_target     = " << cfg.max_macro_steps_target()   << "\n"
	<< "  step_accept_rate_target    = " << cfg.step_accept_rate_target()  << "\n"
	<< "  step_learning_rate         = " << cfg.step_learning_rate()       << "\n"
	<< "  step_gradient_decay        = " << cfg.step_gradient_decay()      << "\n"
	<< "  step_sq_gradient_decay     = " << cfg.step_sq_gradient_decay()   << "\n"
	<< "  step_stabilization         = " << cfg.step_stabilization()       << "\n";
    return out;
  }
  
  class SamplingConfig {
  public:
    uint64_t max_iter()                 const { return max_iter_; }
    uint64_t max_trajectory_doublings() const { return max_trajectory_doublings_; }
    uint64_t max_step_halvings()        const { return max_step_halvings_; }
    double max_hamiltonian_error()      const { return max_hamiltonian_error_; }
    double rhat_converge_tol()          const { return rhat_converge_tol_; }

  private:
    friend class SamplingConfigBuilder;

    SamplingConfig() = default;

    uint64_t max_iter_                 = 1000;
    uint64_t max_trajectory_doublings_ = 5;
    uint64_t max_step_halvings_        = 5;
    double max_hamiltonian_error_      = 0.5;
    double rhat_converge_tol_          = 1.01;
  };

  class SamplingConfigBuilder {
  public:
    SamplingConfigBuilder& max_iter(uint64_t v) {
      validate_finite_positive(v, "max_iter");
      cfg_.max_iter_ = v;
      return *this;
    }
    SamplingConfigBuilder& max_trajectory_doublings(uint64_t v) {
      cfg_.max_trajectory_doublings_ = v;
      return *this;
    }
    SamplingConfigBuilder& max_step_halvings(uint64_t v) {
      cfg_.max_step_halvings_ = v;
      return *this;
    }
    SamplingConfigBuilder& max_hamiltonian_error(double v) {
      validate_finite_positive(v, "max_hamiltonian_error");
      cfg_.max_hamiltonian_error_ = v;
      return *this;
    }
    SamplingConfigBuilder& rhat_converge_tol(double v) {
      validate_finite_gt1(v, "rhat_convergence_tol");
      cfg_.rhat_converge_tol_ = v;
      return *this;
    }

    SamplingConfig build() { return cfg_; }

  private:
    SamplingConfig cfg_;
  };

  std::ostream& operator<<(std::ostream& out, const SamplingConfig& cfg) {
    out << "SamplingConfig\n"
	<< "  max_iter                  = " << cfg.max_iter()                 << "\n"
	<< "  max_trajectory_doublings  = " << cfg.max_trajectory_doublings() << "\n"
	<< "  max_step_halvings         = " << cfg.max_step_halvings()        << "\n"
	<< "  max_hamiltonian_error     = " << cfg.max_hamiltonian_error()    << "\n"
	<< "  rhat_converge_tol         = " << cfg.rhat_converge_tol()        << "\n";
    return out;
  }
  
}  // namespace walnuts
