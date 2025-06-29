#pragma once

#include <utility>

#include "dual_average.hpp"
#include "online_moments.hpp"
#include "util.hpp"
#include "walnuts.hpp"

namespace nuts {

/**
 * @brief Return the gradient of the log density at the specified position.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the target log density/gradient function.
 * @param[in] logp_grad_fun The target log density/gradient function.
 * @param[in] theta The position at which to evaluate the gradient.
 * @return The gradient of the log density at `theta`.
 */
template <typename S, class F>
Vec<S> grad(const F& logp_grad_fun, const Vec<S>& theta) {
  Vec<S> g;
  S logp;
  logp_grad_fun(theta, logp, g);
  return g;
}

/**
 * @brief The immutable mass adaptation configuration.
 *
 * The mass adaptation configuration consists of an initializer for
 * the mass matrix, an initial pseudocount of observations to weight
 * the initial mass matrix (higher values imply more weight on the
 * initial mass matrix), slower updating from the initialization, and
 * an iteration offset for smoothing updates (higher values imply
 * slower updating from the initialization).
 *
 * @tparam S The type of scalars.
 */
template <typename S>
struct MassAdaptConfig {
  /**
   * @brief Construct a mass adaptation configuration with the specified
   * tuning parameters.
   *
   * @param[in] mass_init The diagonal of the diagonal initial mass matrix.
   * @param[in] init_count The pseudocount of observations for the
   * initialization.
   * @param[in] iter_offset The offset from 1 of the first observation.
   * @param[in] additive_smoothing The additive smoothing of inverse mass
   * estimates.
   * @throw std::invalid_argument If the elements of `mass_init` are not finite
   * and positive.
   * @throw std::invalid_argument If the initial count is not finite and
   * positive.
   * @throw std::invalid_argument If the iteration offset is not finite and
   * positive.
   * @throw std::invalid_argument If the additive smoothing is not in (0, 1).
   */
  MassAdaptConfig(const Vec<S>& mass_init, S init_count, S iter_offset,
                  S additive_smoothing)
      : mass_init_(mass_init),
        init_count_(init_count),
        iter_offset_(iter_offset),
        additive_smoothing_(additive_smoothing) {
    if (!(mass_init.array() > 0.0).all() && mass_init.allFinite()) {
      throw std::invalid_argument(
          "Mass matrix entries must be positive finite.");
    }
    if (!(init_count > 0) || std::isinf(init_count)) {
      throw std::invalid_argument("Initial count must be positive finite.");
    }
    if (!(iter_offset > 0) || std::isinf(iter_offset)) {
      throw std::invalid_argument("Iteration offset must be positive finite.");
    }
    if (!(iter_offset > 0) || std::isinf(iter_offset)) {
      throw std::invalid_argument("Iteration offset must be positive finite.");
    }
    if (!(additive_smoothing > 0 && additive_smoothing < 1)) {
      throw std::invalid_argument("Additive smoothing must be in (0, 1).");
    }
  }

  /** The diagonal of the diagonal initial mass matrix. */
  const Vec<S> mass_init_;

  /** The pseudocount for the initial mass matrix. */
  const S init_count_;

  /** The offset from 1 of the first observation. */
  const S iter_offset_;

  /** The additive smoothing. */
  const S additive_smoothing_;
};

/**
 * @brief The immutable configuration for step-size adaptation.
 *
 * The tuning parameters include a step size initialization, a target
 * macro step size bidirectional minimum acceptance rate of the macro step,
 * an iteration offset for smoothing updates (higher is slower to move
 * away from initialization), a learning rate, and decay rate.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
struct StepAdaptConfig {
  /**
   * @brief Construct a step-size adaptation configuration given the
   * tuning parameters.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] accept_rate_target The target bidirectional accept rate.
   * @param[in] iter_offset The relative postion of the first observation.
   * @param[in] learning_rate The learning rate for dual averaging.
   * @param[in] decay_rate The decay rate of older observations.
   * @throw std::invalid_argument If the initial step size is not finite and
   * positive.
   * @throw std::invalid_argument If the acceptance rate target is not in (0,
   * 1).
   * @throw std::invalid_argument If the iteration offset is not finite and
   * positive.
   * @throw std::invalid_argument If the learning rate is not finite and
   * positive.
   * @throw std::invalid_argument If the decay rate is not in (0, 1).
   */
  StepAdaptConfig(S step_size_init, S accept_rate_target, S iter_offset,
                  S learning_rate, S decay_rate)
      : step_size_init_(step_size_init),
        accept_rate_target_(accept_rate_target),
        iter_offset_(iter_offset),
        learning_rate_(learning_rate),
        decay_rate_(decay_rate) {
    if (!(step_size_init > 0) || std::isinf(step_size_init)) {
      throw std::invalid_argument("Initial count must be positive and finite.");
    }
    if (!(accept_rate_target > 0) || !(accept_rate_target < 1)) {
      throw std::invalid_argument("Acceptance rate target must be in (0, 1)");
    }
    if (!(iter_offset > 0) || std::isinf(iter_offset)) {
      throw std::invalid_argument(
          "Iteration offset must be positive and finite.");
    }
    if (!(learning_rate > 0) || std::isinf(learning_rate)) {
      throw std::invalid_argument("Learning rate must be positive and finite.");
    }
    if (!(decay_rate > 0) || !(decay_rate < 1)) {
      throw std::invalid_argument("Decay rate must be in (0, 1)");
    }
  }

  /** The initial macro step size. */
  const S step_size_init_;

  /** The target minimum bidirectional acceptance rate of macro steps. */
  const S accept_rate_target_;

  /** Offset count for initial observation. */
  const S iter_offset_;

  /** Learning rate for dual averaging. */
  const S learning_rate_;

  /** Decay rate for dual averaging. */
  const S decay_rate_;
};

/**
 * @brief The immutable top-level configuration for WALNUTS.
 *
 * A configuration includes a maximum Hamiltonian error per macro leapfrog
 * step, a maximum number of doublings for NUTS, and a maximum number
 * of doublings for number of micro steps.
 *
 * If the maximum number of doublings for NUTS is set to 1, the
 * algorithm reduces to Metropolis-adjusted Langevin (MALA, i.e.,
 * one-step HMC).  If the maximum number of step doublings is set to
 * 0, the algorithm reduces to classical NUTS.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
struct WalnutsConfig {
  /**
   * @brief Construct a WALNUTS configuration with the specified
   * tuning parameters.
   *
   * @param[in] log_max_error The maximum Hamiltonian error per
   * macro leapfrog step.
   * @param[in] max_nuts_depth The maximum number of trajectory
   * doublings for NUTS.
   * @param[in] max_step_depth The maximum number of step doublings
   * per macro step.
   * @throw std::invalid_argument If the log max error is not finite and
   * positive.
   * @throw std::invalid_argument If the maximum tree depth is not positive.
   * @throw std::invalid_argument If the maximum step depth is negative.
   */
  WalnutsConfig(S log_max_error, Integer max_nuts_depth, Integer max_step_depth)
      : log_max_error_(log_max_error),
        max_nuts_depth_(max_nuts_depth),
        max_step_depth_(max_step_depth) {
    if (!(log_max_error > 0) || std::isinf(log_max_error)) {
      throw std::invalid_argument(
          "Log maximum error must be positive and finite.");
    }
    if (max_nuts_depth < 1) {
      throw std::invalid_argument("Maximum NUTS depth must be positive.");
    }
    if (max_step_depth < 0) {
      throw std::invalid_argument("Maximum step depth must be non-negative.");
    }
  }

  /** The maximum error in Hamiltonian in macro steps. */
  const S log_max_error_;

  /** The maximum number of trajectory doublings in NUTS. */
  const Integer max_nuts_depth_;

  /** The maximum number of step doublings per macro step. */
  const Integer max_step_depth_;
};

/**
 * @brief The step-size adaptation handler for WALNUTS.
 *
 * WALNUTS works through callbacks to an adaptation handler, implemented
 * as a functor through the method `operator()(S)`.  This handler maintains
 * the dual averaging adaptation and also returns the current step size
 * estimate through the method `step_size()`.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class StepAdaptHandler {
 public:
  /**
   * Construct a step-size adaptation handler for WALNUTS.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] target_accept_rate The target acceptance rate.
   * @param[in] iter_offset The iteration offset.
   * @param[in] learning_rate The learning rate.
   * @param[in] decay_rate The decay rate.
   * @throw std::invalid_argument If the initial step size is not positive and
   * finite.
   * @throw std::invalid_argument If the target acceptance rate is not in (0,
   * 1).
   * @throw std::invalid_argument If the iteration offset is negative.
   * @throw std::invalid_argument If the learning rate is not positive and
   * finite.
   * @throw std::invalid_argument If the decay rate is not in (0, 1).
   */
  StepAdaptHandler(S step_size_init, S target_accept_rate, S iter_offset,
                   S learning_rate, S decay_rate)
      : dual_average_(step_size_init, target_accept_rate, iter_offset,
                      learning_rate, decay_rate) {
    if (!(step_size_init > 0) || std::isinf(step_size_init)) {
      throw std::invalid_argument("Initial count must be positive finite.");
    }
    if (!(target_accept_rate > 0) || !(target_accept_rate < 1)) {
      throw std::invalid_argument("Target accept rate must be in (0, 1)");
    }
    if (!(decay_rate > 0) || !(decay_rate < 1)) {
      throw std::invalid_argument("Decay rate must be in (0, 1)");
    }
    if (!(iter_offset > 0) || std::isinf(iter_offset)) {
      throw std::invalid_argument(
          "Iteration offset must be positive and finite.");
    }
    if (!(learning_rate > 0) || std::isinf(learning_rate)) {
      throw std::invalid_argument("Learning rate must be positive and finite.");
    }
    if (!(decay_rate > 0) || !(decay_rate < 1)) {
      throw std::invalid_argument("Decay rate must be in (0, 1)");
    }
  }

  /**
   * @brief Update with the estimate of step size given the specified
   * acceptance probability.
   *
   * @param[in] accept_prob The observed acceptance probability.
   */
  void operator()(S accept_prob) { dual_average_.observe(accept_prob); }

  /**
   * @brief Return the estimated step size.
   *
   * @return The estimated step size.
   */
  S step_size() const noexcept { return dual_average_.step_size(); }

 private:
  /** The dual averaging object used for adaptation. */
  DualAverage<S> dual_average_;
};

/**
 * @brief A mass matrix estimator based on exponentially discounted draws
 * and scores (gradients of log densities).
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class MassEstimator {
 public:
  /**
   * @brief Construct a mass matrix estimator with the specified configuration,
   * at the specified initial position and gradient of the log density at the
   * position.
   *
   * The estimator observes positions and their gradients at given iterations
   * with the function `observe()`.  At each step, the discount factor for
   * discounting past draws the online moment estimators is set to
   * ```
   * discount_factor = 1 - 1 / (iter_offset + iter)
   * ```
   * where `iter_offset` is the offset specified in the configuration and `iter`
   * is the iteration number for the observation.
   *
   * The final estimate for the inverse mass matrix is given by the geometric
   * mean of the variance of the scores (the inverse variance estimator)
   * and the variance of the draws (the variance estimator).  This estimate
   * is then additively smoothed by multiplying by multiplying by one minus
   * the additive smoothing and adding the additive smoothing.
   *
   * @param[in] mass_cfg The mass matrix adaptation configuration.
   * @param[in] theta The initial position.
   * @param[in] grad The gradient of the target log density at the initial
   * position.
   * @throw std::invalid_argument If the position and gradient are not the same
   * size.
   */
  MassEstimator(const MassAdaptConfig<S>& mass_cfg, const Vec<S>& theta,
                const Vec<S>& grad)
      : mass_cfg_(mass_cfg),
        var_estimator_(0, theta.size()),
        inv_var_estimator_(0, theta.size()) {
    S smoothing = mass_cfg_.additive_smoothing_;
    Vec<S> zero = Vec<S>::Zero(theta.size());
    Vec<S> smooth_vec = Vec<S>::Constant(theta.size(), smoothing);
    Vec<S> sqrt_abs_grad_init = grad.array().abs().sqrt();
    Vec<S> init_prec = (1 - smoothing) * sqrt_abs_grad_init + smooth_vec;
    Vec<S> init_var = init_prec.array().inverse().matrix();
    S dummy_discount = 0.98;  // gets reset before being used
    inv_var_estimator_ = OnlineMoments<S, Integer>(
        dummy_discount, mass_cfg.iter_offset_, zero, init_prec);
    var_estimator_ = OnlineMoments<S, Integer>(
        dummy_discount, mass_cfg.iter_offset_, zero, init_var);
    if (theta.size() != grad.size()) {
      throw std::invalid_argument(
          "Position and gradient must be the same size.");
    }
  }

  /**
   * @brief Update the estimate for the specified iteration with the
   * observation and gradient.
   *
   * @param[in] theta The position observed.
   * @param[in] grad The gradient of the log density at the position.
   * @param[in] iteration The iteration number (non-negative integer).
   * @pre theta.size() = grad.size()
   * @pre iteration >= 0
   */
  void observe(const Vec<S>& theta, const Vec<S>& grad, Integer iteration) {
    double discount_factor = 1.0 - 1.0 / (mass_cfg_.iter_offset_ + iteration);
    var_estimator_.set_discount_factor(
        discount_factor);  // TODO: one encapsulated function
    var_estimator_.observe(theta);
    inv_var_estimator_.set_discount_factor(discount_factor);
    inv_var_estimator_.observe(grad);
  }

  /**
   * @brief Return an estimate of the inverse mass matrix.
   *
   * @return The inverse mass matrix estimate.
   */
  Vec<S> inv_mass_estimate() const {
    Vec<S> inv_mass_est_var = var_estimator_.variance().array();
    Vec<S> inv_mass_est_inv_var =
        inv_var_estimator_.variance().array().inverse().matrix();
    Vec<S> inv_mass_est =
        (inv_mass_est_var.array() * inv_mass_est_inv_var.array())
            .sqrt()
            .matrix();
    return inv_mass_est;
  }

 private:
  /** The mass matrix adaptation configuration. */
  MassAdaptConfig<S> mass_cfg_;

  /** The online variance estimator for draws. */
  OnlineMoments<S, Integer> var_estimator_;

  /** The online inverse variance estimator for scores. */
  OnlineMoments<S, Integer> inv_var_estimator_;
};

/**
 * @brief The adaptive WALNUTS sampler.
 *
 * The adaptive WALNUTS sampler is configured in the constructor, then
 * provides a functor method `operator()()` for returning the next
 * state in warmup.  Warmup can be called externally to be run for any
 * number of iterations.  Warmup continually re-estimates step size
 * and mass matrix each iteration, exponentially discounting the past.
 *
 * After adaptation, the method `sampler()` returns the a WALNUTS
 * sampler configured with the result of adaptation.
 *
 * The target log density and gradient function must implement the signature
 *
 * ```cpp
 * static void normal_logp_grad(const Eigen::Matrix<S, -1, 1>& x,
 *                              S& logp,
 *                              Eigen::Matrix<S, -1, 1>& grad);
 * ```
 *
 * where `S` is the scalar type parameter of the sampler (the log
 * density function need not be templated itself.  The argument `x`
 * is the position argument, and `logp` is set to the log density of
 * `x`, and `grad` set to the gradient of the log density at `x`.
 *
 * @tparam F Type of log density/gradient function.
 * @tparam S Type of scalars.
 * @tparam RNG Type of base random number generator.
 */
template <class F, typename S, class RNG>
class AdaptiveWalnuts {
 public:
  /**
   * @brief Construct an adaptive WALNUTS sampler.
   *
   * The configuration objects are moved, the initialization is
   * copied, and the base random number generator and log
   * density/gradient function are held by reference.  The RNG is
   * changes every time a random number is generated.  The target log
   * density can be non-constant to allow for implementations that,
   * for example, track number of gradient evaluations, but nothing
   * is done by WALNUTS to mutate it.
   *
   * @param[in,out] rng The base random number generator.
   * @param[in,out] logp_grad The target log density and gradient function.
   * @param[in] theta_init The initial state.
   * @param[in] mass_cfg The mass-matrix adaptation configuration.
   * @param[in] step_cfg The step-size adaptation configuration.
   * @param[in] walnuts_cfg The WALNUTS configuration.
   */
  AdaptiveWalnuts(RNG& rng, F& logp_grad, const Vec<S>& theta_init,
                  const MassAdaptConfig<S>& mass_cfg,
                  const StepAdaptConfig<S>& step_cfg,
                  const WalnutsConfig<S>& walnuts_cfg)
      : mass_cfg_(mass_cfg),
        step_cfg_(step_cfg),
        walnuts_cfg_(walnuts_cfg),
        rand_(rng),
        logp_grad_(logp_grad),
        theta_(theta_init),
        iteration_(0),
        step_adapt_handler_(step_cfg.step_size_init_,
                            step_cfg.accept_rate_target_, step_cfg.iter_offset_,
                            step_cfg.learning_rate_, step_cfg.decay_rate_),
        mass_estimator_(mass_cfg_, theta_, grad(logp_grad, theta_)) {}

  /**
   * @brief Return the next state from warmup.
   *
   * This method should be called a number of time equal to the
   * number of warmup iterations desired.  These warmup draws are
   * *not* drawn from a Markov chain and are not valid for inference.
   * After warmup, call `sampler()` to return a sampler that fixes the
   * tuning parameters and provides a proper Markov chain.
   *
   * @return The next warmup state.
   */
  const Vec<S> operator()() {
    Vec<S> inv_mass = mass_estimator_.inv_mass_estimate();
    Vec<S> chol_mass = inv_mass.array().inverse().sqrt().matrix();
    Vec<S> grad_select;
    theta_ = transition_w(
        rand_, logp_grad_, inv_mass, chol_mass, step_adapt_handler_.step_size(),
        walnuts_cfg_.max_nuts_depth_, std::move(theta_), grad_select,
        walnuts_cfg_.log_max_error_, step_adapt_handler_);
    mass_estimator_.observe(theta_, grad_select, iteration_);
    ++iteration_;
    return theta_;
  }

  /**
   * @brief Return a WALNUTS sampler with the current tuning parameter
   * estimates.
   *
   * The returned sampler forms a proper Markov chain.  The method passes
   * along the compound random number generator and log density function and
   * is hence not marked `const`.
   *
   * @return The WALNUTS sampler with current tuning parameter estimates.
   */
  WalnutsSampler<F, S, RNG> sampler() {
    return WalnutsSampler<F, S, RNG>(
        rand_, logp_grad_, theta_, mass_estimator_.inv_mass_estimate(),
        step_adapt_handler_.step_size(), walnuts_cfg_.max_nuts_depth_,
        walnuts_cfg_.log_max_error_);
  }

 private:
  /** The mass adaptation configuration. */
  const MassAdaptConfig<S> mass_cfg_;

  /** The step-size adaptation configuration. */
  const StepAdaptConfig<S> step_cfg_;

  /** The WALNUTS sampler configuration. */
  const WalnutsConfig<S> walnuts_cfg_;

  /** The random number generator required for NUTS. */
  Random<S, RNG> rand_;

  /** The target log density/gradient function. */
  F& logp_grad_;

  /** The current state. */
  Vec<S> theta_;

  /** The current iteration. */
  Integer iteration_;

  /** The handler for WALNUTS for step size adaptation. */
  StepAdaptHandler<S> step_adapt_handler_;

  /** The estimat for mass matrices. */
  MassEstimator<S> mass_estimator_;
};

}  // namespace nuts
