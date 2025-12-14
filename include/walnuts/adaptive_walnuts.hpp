#pragma once

#include <utility>

#include "adam.hpp"
#include "online_moments.hpp"
#include "util.hpp"
#include "walnuts.hpp"

namespace nuts {

/**
 * @brief Return the gradient of the log density at the specified position.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the target log density/gradient function.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] theta The position at which to evaluate the gradient.
 * @return The gradient of the log density at `theta`.
 */
template <typename S, class F>
Vec<S> grad(const F& logp_grad, const Vec<S>& theta) {
  Vec<S> g;
  S logp;
  logp_grad(theta, logp, g);
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
    validate_positive(mass_init, "mass_init entries");
    validate_positive(init_count, "init_count");
    validate_positive(iter_offset, "iter_offset");
    validate_probability(additive_smoothing, "additive_smoothing");
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
   * @param[in] max_error The maximum Hamiltonian error per
   * macro leapfrog step.
   * @param[in] max_nuts_depth The maximum number of trajectory
   * doublings for NUTS.
   * @param[in] max_step_halvings The maximum number of step halvings
   * per macro step.
   * @param[in] min_micro_steps The minimum number of micro steps per macro
   *  step.
   * @throw std::invalid_argument If the max error is not finite and
   * positive.
   * @throw std::invalid_argument If the maximum NUTS depth is zero.
   * @throw std::invalid_argument If the maximum number of step halvings is
   * zero.
   * @throw std::invalid_argument If the minimum number of micro steps is zero.
   */
  WalnutsConfig(S max_error, std::size_t max_nuts_depth,
                std::size_t max_step_halvings, std::size_t min_micro_steps)
      : max_error_(max_error),
        max_nuts_depth_(max_nuts_depth),
        max_step_halvings_(max_step_halvings),
        min_micro_steps_(min_micro_steps) {
    validate_positive(max_error, "max_error");
    validate_positive(max_nuts_depth, "max_nuts_depth");
    validate_positive(max_step_halvings, "max_step_halvings");
    validate_positive(min_micro_steps, "min_micro_steps");
  }

  /** The maximum error in Hamiltonian in macro steps. */
  const S max_error_;

  /** The maximum number of trajectory doublings in NUTS. */
  const std::size_t max_nuts_depth_;

  /** The maximum number of step doublings per macro step. */
  const std::size_t max_step_halvings_;

  /** The minimum number of micro steps per macro step. */
  const std::size_t min_micro_steps_;
};

/**
 * @brief The step-size adaptation handler for WALNUTS.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class StepAdaptHandler {
 public:
  /**
   * Construct a step-size adaptation handler for WALNUTS.
   *
   * @param[in] cfg The stepsize adaptation tuning parameters.
   */
  StepAdaptHandler(const AdamConfig<S>& cfg) : adam_(cfg) {}

  /**
   * @brief Update with the estimate of step size given the specified
   * acceptance probability.
   *
   * @param[in] accept_prob The observed acceptance probability.
   */
  void operator()(S accept_prob) { adam_.observe(accept_prob); }

  /**
   * @brief Return the estimated step size.
   *
   * @return The estimated step size.
   */
  S step_size() const noexcept { return adam_.step_size(); }

 private:
  /** The Adam instance for step size adaptation. */
  Adam<S> adam_;
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
      : mass_cfg_(mass_cfg) {
    // 0.98 is dummy that will get overwritten
    // var_estimator_(0.98, static_cast<std::size_t>(theta.size())),
    // inv_var_estimator_(0.98, static_cast<std::size_t>(theta.size())) {
    validate_same_size(theta, grad, "theta", "grad");

    S smoothing = mass_cfg_.additive_smoothing_;
    Vec<S> zero = Vec<S>::Zero(theta.size());
    Vec<S> smooth_vec = Vec<S>::Constant(theta.size(), smoothing);
    Vec<S> sqrt_abs_grad_init = grad.array().abs().sqrt();
    Vec<S> init_prec = (1 - smoothing) * sqrt_abs_grad_init + smooth_vec;
    Vec<S> init_var = init_prec.array().inverse().matrix();
    S dummy_discount = 0.98;  // gets reset before being used
    inv_var_estimator_ = OnlineMoments<S>(dummy_discount, mass_cfg.iter_offset_,
                                          zero, init_prec);
    var_estimator_ =
        OnlineMoments<S>(dummy_discount, mass_cfg.iter_offset_, zero, init_var);
  }

  /**
   * @brief Update the estimate for the specified iteration with the
   * observation and gradient.
   *
   * @param[in] theta The position observed.
   * @param[in] grad The gradient of the log density at the position.
   * @param[in] iteration The iteration number.
   * @pre theta.size() = grad.size()
   * @pre iteration >= 0
   */
  void observe(const Vec<S>& theta, const Vec<S>& grad, std::size_t iteration) {
    double discount_factor = 1.0 - 1.0 / (mass_cfg_.iter_offset_ + iteration);
    var_estimator_.discount_observe(discount_factor, theta);
    inv_var_estimator_.discount_observe(discount_factor, grad);
  }

  /**
   * @brief Return an estimate of the inverse mass matrix.
   *
   * @return The inverse mass matrix estimate.
   */
  Vec<S> inv_mass_estimate() const {
    return (var_estimator_.variance().array() *
            inv_var_estimator_.variance().array().inverse())
        .sqrt()
        .matrix();
  }

 private:
  /** The mass matrix adaptation configuration. */
  MassAdaptConfig<S> mass_cfg_;

  /** The online variance estimator for draws. */
  OnlineMoments<S> var_estimator_;

  /** The online inverse variance estimator for scores. */
  OnlineMoments<S> inv_var_estimator_;
};

/**
 * @brief The adaptation handler for the minimum number of micro steps per macro step.  
 *
 * After being constructed with a target number of macro steps, this class observes
 * the number of micro steps taken and adjusts the minimum number of micro steps per
 * macro step in order to achieve the target expected number of macro steps.  There is
 * slight regularization toward one, but otherwise it just uses the floor of an average
 * and thus rounds down.
 */
class MinMicroStepsAdaptHandler {
public: 
  /**
   * Construct a minimum number of micro steps per macro step handler.
   *
   * @param[in] expected_macro_steps Expected number of macro steps.  
   */
  MinMicroStepsAdaptHandler(double expected_macro_steps) :
    expected_macro_steps_(expected_macro_steps),
    total_macro_steps_(2.0),
    count_(1.0) {
  }

  /**
   * @brief Observe the specifed number of macro steps in a NUTS trajectory.
   *
   * @param[in] macro_steps The number of macro steps used in a trajectory.
   */
  void observe(std::size_t macro_steps) {
    total_macro_steps_ += static_cast<double>(macro_steps);
    ++count_;
  }

  /**
   * @brief Return the estimated minimum number of micro steps.
   *
   * This estimate is designed to achieve the expected number of macro steps
   * per iteration.
   *
   * @return The minimum number of micro steps to use per macro step. 
   */
  std::size_t min_micro_steps() const noexcept {
    double mean_micro = total_macro_steps_ / count_;
    double min_micro_per_macro
      = std::fmax(1.0, std::floor(mean_micro / expected_macro_steps_));
    std::size_t steps
      = static_cast<std::size_t>(std::round(min_micro_per_macro));
    return steps;
  }
  
private:
  const double expected_macro_steps_;
  double total_macro_steps_;
  double count_;
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
   * for example, track number of gradient evaluations.  The target depth
   * specifies the expected NUTS tree depth, which is controlled through
   * the minimum number of micro steps per macro step and adjusted with
   * a mean estimator to achieve this average.
   *
   * @param[in,out] rng The base random number generator.
   * @param[in] logp_grad The target log density and gradient function.
   * @param[in] theta_init The initial state.
   * @param[in] mass_cfg The mass-matrix adaptation configuration.
   * @param[in] step_cfg The step-size adaptation configuration.
   * @param[in] walnuts_cfg The WALNUTS configuration.
   * @param[in] target_depth The target expected NUTS tree depth.
   */
  AdaptiveWalnuts(RNG& rng, const F& logp_grad, const Vec<S>& theta_init,
                  const MassAdaptConfig<S>& mass_cfg,
                  const AdamConfig<S>& step_cfg,
                  const WalnutsConfig<S>& walnuts_cfg,
		  double target_depth = 4.0)
      : mass_cfg_(mass_cfg),
        step_cfg_(step_cfg),
        walnuts_cfg_(walnuts_cfg),
        rand_(rng),
        logp_grad_(logp_grad),
        theta_(theta_init),
        iteration_(0),
        step_adapt_handler_(step_cfg),
        mass_estimator_(mass_cfg_, theta_, grad(logp_grad, theta_)),
	min_micro_estimator_(target_depth) {
  }

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
    std::size_t depth = 0;
    theta_ = transition_w(
        rand_, logp_grad_, inv_mass, chol_mass, step_adapt_handler_.step_size(),
        walnuts_cfg_.max_nuts_depth_, walnuts_cfg_.max_step_halvings_,
        min_micro_estimator_.min_micro_steps(), walnuts_cfg_.max_error_,
        std::move(theta_), depth, grad_select, step_adapt_handler_);
    mass_estimator_.observe(theta_, grad_select, iteration_);
    min_micro_estimator_.observe(depth);
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
        rand_, logp_grad_.logp_grad_, theta_,
        mass_estimator_.inv_mass_estimate(), step_adapt_handler_.step_size(),
        walnuts_cfg_.max_nuts_depth_, walnuts_cfg_.max_step_halvings_,
        min_micro_estimator_.min_micro_steps(), walnuts_cfg_.max_error_);
  }

  /**
   * @brief Return the diagonal of the diagonal inverse mass matrix.
   *
   * @return The diagonal of the inverse mass matrix.
   */
  Vec<S> inv_mass() const { return mass_estimator_.inv_mass_estimate(); }

  /**
   * @brief Return the step size.
   *
   * @return The step size.
   */
  S step_size() const { return step_adapt_handler_.step_size(); }

  /**
   * @brief Return the minimum number of micro steps per macro step.
   *
   * @return The minimum number of micro steps per macro step.
   */
  std::size_t min_micro_steps() const {
    return  min_micro_estimator_.min_micro_steps();
  }
  
 private:
  /** The mass adaptation configuration. */
  const MassAdaptConfig<S> mass_cfg_;

  /** The step-size adaptation configuration. */
  const AdamConfig<S> step_cfg_;

  /** The WALNUTS sampler configuration. */
  const WalnutsConfig<S> walnuts_cfg_;

  /** The random number generator required for NUTS. */
  Random<S, RNG> rand_;

  /** The target log density/gradient function. */
  const NoExceptLogpGrad<F, S> logp_grad_;

  /** The current state. */
  Vec<S> theta_;

  /** The current iteration. */
  std::size_t iteration_;

  /** The handler receiving observations from WALNUTS for step size adaptation.
   */
  StepAdaptHandler<S> step_adapt_handler_;

  /** The estimat for mass matrices. */
  MassEstimator<S> mass_estimator_;

  /** The estimator for the minimum number of micro steps per macro step. */
  MinMicroStepsAdaptHandler min_micro_estimator_;
};

}  // namespace nuts
