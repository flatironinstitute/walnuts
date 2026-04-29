#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include <walnuts/config.hpp>
#include <walnuts/util.hpp>
#include <walnuts/validate.hpp>

namespace walnuts {

/**
 * @brief The initialization configuration for a single Markov chain.
 *
 * The initialization configuration specifies a step size, initial position,
 * and initial mass matrix.
 */
class InitChainConfig {
 public:
  /**
   * @brief Construct an initialization configuration.
   *
   * @param step_size The initial step size.
   * @param position The initial position.
   * @param mass The initial mass matrix (diagonal).
   */
  InitChainConfig(double step_size, const Eigen::VectorXd& position,
                  const Eigen::VectorXd& mass)
      : step_size_(step_size), position_(position), mass_(mass) {}

  /**
   * @brief Return the initial step size.
   *
   * @return The step size.
   */
  double step_size() const noexcept { return step_size_; }

  /**
   * @brief Return the initial position.
   *
   * @return The position.
   */
  const Eigen::VectorXd& position() const noexcept { return position_; }

  /**
   * @brief Return the initial mass matrix.
   *
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass() const noexcept { return mass_; }

 private:
  const double step_size_;
  const Eigen::VectorXd position_;
  const Eigen::VectorXd mass_;
};

/**
 * The initialization configuration for multiple Markov chains.
 * Rather than a public constructor, it is built using an
 * `InitConfigBuilder` instance.
 *
 * The initialization configuration specifies a step size, initial
 * position, and initial mass matrix.
 */
class InitConfig {
 public:
  /**
   * @brief Return the number of chains.
   *
   * @return The number of chains.
   */
  uint64_t num_chains() const noexcept { return step_sizes_.size(); }

  /**
   * @brief Return the dimensionality of the positions.
   *
   * If the initialization is empty, 0 is returned.
   *
   * @return The dimensionality.
   */
  uint64_t dims() const noexcept {
    return positions_.empty()
               ? 0u
               : static_cast<uint64_t>(positions_.front().size());
  }

  /**
   * @brief Return the initial step sizes.
   *
   * @return The step sizes.
   */
  const std::vector<double>& step_sizes() const noexcept { return step_sizes_; }

  /**
   * @brief Return the initial step size for the specified chain.
   *
   * @param n The chain identifier.
   * @return The step size.
   */
  double step_size(std::size_t n) const noexcept { return step_sizes_[n]; }

  /**
   * @brief Return the initial positions for all chains.
   *
   * @return The positions.
   */
  const std::vector<Eigen::VectorXd>& positions() const noexcept {
    return positions_;
  }

  /**
   * @brief Return the initial position for the specified chain.
   *
   * @param n The chain index.
   * @return The positions.
   */
  const Eigen::VectorXd& position(std::size_t n) const noexcept {
    return positions_[n];
  }

  /**
   * @brief Return the initial diagonal mass matrices for all chains.
   *
   * @return The mass matrix diagonals.
   */
  const std::vector<Eigen::VectorXd>& masses() const noexcept {
    return masses_;
  }

  /**
   * @brief Return the mass matrix for the specified chain.
   *
   * @param n The chain index.
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass(std::size_t n) const noexcept {
    return masses_[n];
  }

  /**
   * @brief Return the initialization configuration for the specified chain.
   *
   * @param n The chain index.
   * @return The indexed chain's initialization configuration.
   */
  InitChainConfig init_chain_config(std::size_t n) const {
    return InitChainConfig(step_size(n), position(n), mass(n));
  }

 private:
  friend class InitConfigBuilder;

  InitConfig() = default;

  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;
};

/**
 * @brief The builder for initialization configurations.
 *
 * The usage to return an `InitChainConfig` is `InitConfigBuilder(4,
 * 20).step_sizes(0.5).build();` with any number of config methods
 * chained between the construction and call to build.
 */
class InitConfigBuilder {
 public:
  /**
   * @brief Construct an initialization builder of the given sizes.
   *
   * @param num_chains The number of Markov chains.
   * @param dims The dimensionality of each chain.
   */
  InitConfigBuilder(uint64_t num_chains, uint64_t dims)
      : num_chains_(num_chains), dims_(dims) {
    this->step_sizes(0.1);
    Eigen::VectorXd position =
        Eigen::VectorXd::Zero(static_cast<int64_t>(dims));
    this->positions(position);
    Eigen::VectorXd mass = Eigen::VectorXd::Ones(static_cast<int64_t>(dims));
    this->masses(mass);
  }

  /**
   * @brief Set the step sizes to all be the specified value.
   *
   * @param v The step size.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the step size is not finite and positive.
   */
  InitConfigBuilder& step_sizes(double v) {
    validate_finite_positive(v, "step size");
    step_sizes_ = std::vector<double>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Set the step sizes to all be the specified values.
   *
   * @param v The step sizes.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If any of the step sizes are not finite
   * positive.
   * @throw std::invalid_argument If the number of chains doesn't match the
   * number specified in the constuctor.
   */
  InitConfigBuilder& step_sizes(const std::vector<double>& v) {
    validate_size(v, num_chains_, "step_sizes", "num_chains");
    validate_finite_positive(v, "step_size");
    step_sizes_ = v;
    return *this;
  }

  /**
   * @brief Randomly initialization the positions.
   *
   * Initialization is independent in each dimension with values drawn
   * from a zero-centered normal distribution with the specified
   * scale.
   *
   * @tparam RNG The type of the base random number generator.
   * @param init_scale The scale of the normal initial values.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the initial scale is not finite and
   * positive.
   */
  template <std::uniform_random_bit_generator RNG>
  InitConfigBuilder& positions(RNG& rng, double init_scale) {
    validate_finite_positive(init_scale, "init_scale");
    Random<double, RNG> rand(rng);
    positions_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      rand.standard_normal(dims_, positions_[c]);
      positions_[c] *= init_scale;
    }
    return *this;
  }

  /**
   * @brief Initialize the positions all to the same value.
   *
   * @param v The initial position.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   */
  InitConfigBuilder& positions(const Eigen::VectorXd& v) {
    validate_size(v, dims_, "position", "dims");
    validate_finite(v, "position");
    positions_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values.
   *
   * @param vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(const std::vector<Eigen::VectorXd>& vs) {
    validate_size(vs, num_chains_, "positions", "num_chains");
    validate_finite(vs, "positions");
    for (const auto& v : vs) {
      validate_size(v, dims_, "position", "dims");
    }
    positions_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values via move.
   *
   * @param vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(std::vector<Eigen::VectorXd>&& vs) {
    validate_size(vs, num_chains_, "positions", "num_chains");
    validate_finite(vs, "positions");
    for (const auto& v : vs) {
      validate_size(v, dims_, "position", "dims");
    }
    positions_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Initialize the masses using the Nutpie outer product strategy.
   *
   * The initialization uses a smoothed negative outer product of
   * gradients, following Nutpie (Seyboldt et al. 2026 \cite
   * seyboldt2025nutpie).  More specifically, it uses the square root
   * of the absolute value of the outer proudct of gradients linearly
   * interpolated with a unit matrix with weight `mass_smoothing` on
   * the unit matrix and `1 - mass_smoothing` on the regularized outer
   * product.  This regularization goes beyond Nutpie to do the linear
   * interpolation and also to take a square root to further
   * regularize.
   *
   * @tparam LPG The type of the log density and gradient function.
   * @param logp_grad The log density and gradient function.
   * @param mass_smoothing The additive smoothing for mass matrices.
   * @throw std::invalid_argumet If the mass smoothing is not in (0, 1).
   * @return A reference to this builder for chaining.
   */
  template <typename LPG>
  InitConfigBuilder& masses(const LPG& logp_grad, double mass_smoothing) {
    validate_probability(mass_smoothing, "mass_smoothing");
    Eigen::VectorXd grad;
    double lp;  // needed to calculate gradient, o.w. not used
    masses_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      logp_grad(positions_[c], lp, grad);
      // abs geom averages with unit, sqrt additionally regularizes scale
      masses_[c] =
          (1 - mass_smoothing) * grad.array().abs().sqrt() + mass_smoothing;
    }
    return *this;
  }

  /**
   * @brief Initialize the mass matrices all to the same value.
   *
   * @param v The initial diagonal mass matrix.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial mass matrix
   * diagonals contains non-finite or non-positive values.
   */
  InitConfigBuilder& masses(const Eigen::VectorXd& v) {
    validate_size(v, dims_, "masses", "dims");
    validate_finite_positive(v, "masses");
    masses_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values.
   *
   * @param vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(const std::vector<Eigen::VectorXd>& vs) {
    validate_size(vs, num_chains_, "masses", "num_chains");
    validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values via
   * move.
   *
   * @param vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(std::vector<Eigen::VectorXd>&& vs) {
    validate_size(vs, num_chains_, "masses", "num_chains");
    validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Return the initialization configuration.
   *
   * @return The initialization configuration.
   */
  InitConfig build() {
    InitConfig cfg;
    cfg.step_sizes_ = std::move(step_sizes_);
    cfg.positions_ = std::move(positions_);
    cfg.masses_ = std::move(masses_);
    return cfg;
  }

 private:
  uint64_t num_chains_;
  uint64_t dims_;
  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;
};

/**
 * Write a dump of the initial configurations to the specified stream.
 *
 * @param out Stream to which configuration is written.
 * @param cfg The configuration to write.
 * @return A reference to the output stream for chaining.
 */
inline std::ostream& operator<<(std::ostream& out, const InitConfig& cfg) {
  out << "InitConfigs (by chain)\n";
  for (std::size_t n = 0; n < cfg.step_sizes().size(); ++n) {
    if (n > 0) {
      out << "\n";
    }
    out << "  chain         = " << n << "\n"
        << "    num_chains  = " << cfg.num_chains() << "\n"
        << "    step_size   = " << cfg.step_sizes()[n] << "\n"
        << "    position    = " << cfg.positions()[n].transpose() << "\n"
        << "    mass        = " << cfg.masses()[n].transpose() << "\n";
  }
  return out;
}

class WarmupConfig {
 public:
  uint64_t min_iter() const { return min_iter_; }
  uint64_t max_iter() const { return max_iter_; }
  double step_size_converge_tol() const { return step_size_converge_tol_; }
  double mass_converge_tol() const { return mass_converge_tol_; }
  double mass_init_count() const { return mass_init_count_; }
  double mass_additive_smoothing() const { return mass_additive_smoothing_; }
  double max_macro_steps_target() const { return max_macro_steps_target_; }
  double step_accept_rate_target() const { return step_accept_rate_target_; }
  double step_learning_rate() const { return step_learning_rate_; }
  double step_gradient_decay() const { return step_gradient_decay_; }
  double step_sq_gradient_decay() const { return step_sq_gradient_decay_; }
  double step_stabilization() const { return step_stabilization_; }
  double step_learn_rate_decay() const { return step_learn_rate_decay_; }
  uint64_t publish_stride() const { return publish_stride_; }
  uint64_t probe_microseconds() const { return probe_microseconds_; }
  uint64_t yield_period() const { return yield_period_; }

 private:
  friend class WarmupConfigBuilder;

  WarmupConfig() = default;

  uint64_t min_iter_ = 50;
  uint64_t max_iter_ = 1000;
  double step_size_converge_tol_ = 0.1;
  double mass_converge_tol_ = 1.0;
  double mass_init_count_ = 4.0;
  double mass_additive_smoothing_ = 1e-5;
  double max_macro_steps_target_ = 15.0;
  double step_accept_rate_target_ = 0.8;
  double step_learning_rate_ = 0.05;
  double step_gradient_decay_ = 0.8;
  double step_sq_gradient_decay_ = 0.9;
  double step_stabilization_ = 1e-4;
  double step_learn_rate_decay_ = 0.5;
  uint64_t publish_stride_ = 5;
  uint64_t probe_microseconds_ = 1000;
  uint64_t yield_period_ = 32;
};

class WarmupConfigBuilder {
 public:
  WarmupConfigBuilder& min_max_iter(uint64_t min_iter, uint64_t max_iter) {
    if (!(min_iter <= max_iter)) {
      throw std::invalid_argument(
          "min_iter cannot be greater than than max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
    return *this;
  }
  WarmupConfigBuilder& step_size_converge_tol(double v) {
    validate_finite_positive(v, "step_size_converge_tol");
    cfg_.step_size_converge_tol_ = v;
    return *this;
  }
  WarmupConfigBuilder& mass_converge_tol(double v) {
    validate_finite_positive(v, "mass_converge_tol");
    cfg_.mass_converge_tol_ = v;
    return *this;
  }
  WarmupConfigBuilder& mass_init_count(double v) {
    validate_finite_positive(v, "mass_init_count");
    cfg_.mass_init_count_ = v;
    return *this;
  }
  WarmupConfigBuilder& mass_additive_smoothing(double v) {
    validate_finite_positive(v, "mass_additive_smoothing");
    cfg_.mass_additive_smoothing_ = v;
    return *this;
  }
  WarmupConfigBuilder& max_macro_steps_target(double v) {
    validate_finite_positive(v, "max_macro_steps_target");
    cfg_.max_macro_steps_target_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_accept_rate_target(double v) {
    validate_probability(v, "step_accept_rate_target");
    cfg_.step_accept_rate_target_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_learning_rate(double v) {
    validate_finite_positive(v, "step_learning_rate");
    cfg_.step_learning_rate_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_gradient_decay(double v) {
    validate_probability(v, "step_gradient_decay");
    cfg_.step_gradient_decay_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_sq_gradient_decay(double v) {
    validate_probability(v, "step_sq_gradient_decay");
    cfg_.step_sq_gradient_decay_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_stabilization(double v) {
    validate_finite_positive(v, "step_stabilization");
    cfg_.step_stabilization_ = v;
    return *this;
  }
  WarmupConfigBuilder& step_learn_rate_decay(double v) {
    validate_probability(v, "step_learn_rate_decay");
    cfg_.step_learn_rate_decay_ = v;
    return *this;
  }
  WarmupConfigBuilder& publish_stride(uint64_t v) {
    validate_positive(v, "publish_stride");
    cfg_.publish_stride_ = v;
    return *this;
  }
  WarmupConfigBuilder& probe_microseconds(uint64_t v) {
    validate_positive(v, "probe_microseconds");
    cfg_.probe_microseconds_ = v;
    return *this;
  }
  WarmupConfigBuilder& yield_period(uint64_t v) {
    validate_positive(v, "yield_period");
    cfg_.yield_period_ = v;
    return *this;
  }

  WarmupConfig build() { return cfg_; }

 private:
  WarmupConfig cfg_;
};

inline std::ostream& operator<<(std::ostream& out, const WarmupConfig& cfg) {
  out << "WarmupConfig\n"
      << "  min_iter                 = " << cfg.min_iter() << "\n"
      << "  max_iter                 = " << cfg.max_iter() << "\n"
      << "  step_size_converge_tol   = " << cfg.step_size_converge_tol() << "\n"
      << "  mass_converge_tol        = " << cfg.mass_converge_tol() << "\n"
      << "  mass_init_count          = " << cfg.mass_init_count() << "\n"
      << "  mass_additive_smoothing  = " << cfg.mass_additive_smoothing()
      << "\n"
      << "  max_macro_steps_target   = " << cfg.max_macro_steps_target() << "\n"
      << "  step_accept_rate_target  = " << cfg.step_accept_rate_target()
      << "\n"
      << "  step_learning_rate       = " << cfg.step_learning_rate() << "\n"
      << "  step_gradient_decay      = " << cfg.step_gradient_decay() << "\n"
      << "  step_sq_gradient_decay   = " << cfg.step_sq_gradient_decay() << "\n"
      << "  step_stabilization       = " << cfg.step_stabilization() << "\n"
      << "  step_learn_rate_decay    = " << cfg.step_learn_rate_decay() << "\n"
      << "  publish_stride           = " << cfg.publish_stride() << "\n"
      << "  probe_microseconds       = " << cfg.probe_microseconds() << "\n"
      << "  yield_period             = " << cfg.yield_period() << "\n";
  return out;
}

class SamplingConfig {
 public:
  uint64_t min_iter() const noexcept { return min_iter_; }
  uint64_t max_iter() const noexcept { return max_iter_; }
  uint64_t max_trajectory_doublings() const noexcept {
    return max_trajectory_doublings_;
  }
  uint64_t max_step_halvings() const noexcept { return max_step_halvings_; }
  double max_hamiltonian_error() const noexcept {
    return max_hamiltonian_error_;
  }
  uint64_t min_micro_steps() const noexcept { return min_micro_steps_; }
  double rhat_converge_tol() const noexcept { return rhat_converge_tol_; }

 private:
  friend class SamplingConfigBuilder;

  SamplingConfig() = default;

  uint64_t min_iter_ = 50;
  uint64_t max_iter_ = 1000;
  uint64_t max_trajectory_doublings_ = 5;
  uint64_t max_step_halvings_ = 5;
  double max_hamiltonian_error_ = 0.5;
  uint64_t min_micro_steps_ = 1;
  double rhat_converge_tol_ = 1.01;
};

class SamplingConfigBuilder {
 public:
  SamplingConfigBuilder& min_max_iter(uint64_t min_iter, uint64_t max_iter) {
    if (!(min_iter <= max_iter)) {
      throw std::invalid_argument("min_iter must be <= max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
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
  SamplingConfigBuilder& min_micro_steps(uint64_t v) {
    validate_positive(v, "min_micro_steps");
    cfg_.min_micro_steps_ = v;
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

inline std::ostream& operator<<(std::ostream& out, const SamplingConfig& cfg) {
  out << "SamplingConfig\n"
      << "  min_iter                   = " << cfg.min_iter() << "\n"
      << "  max_iter                   = " << cfg.max_iter() << "\n"
      << "  max_trajectory_doublings   = " << cfg.max_trajectory_doublings()
      << "\n"
      << "  max_step_halvings          = " << cfg.max_step_halvings() << "\n"
      << "  max_hamiltonian_error      = " << cfg.max_hamiltonian_error()
      << "\n"
      << "  min_micro_steps            = " << cfg.min_micro_steps() << "\n"
      << "  rhat_converge_tol          = " << cfg.rhat_converge_tol() << "\n";
  return out;
}

}  // namespace walnuts
