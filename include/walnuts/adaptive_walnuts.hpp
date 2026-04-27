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
 * @brief The step-size adaptation handler for WALNUTS.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class StepAdaptHandler {
 public:
  StepAdaptHandler(const walnuts::InitChainConfig& init_chain_cfg,
		   const walnuts::WarmupConfig& warmup_cfg)
    : adam_(init_chain_cfg.step_size(),
	    warmup_cfg.step_accept_rate_target(),
	    warmup_cfg.step_learning_rate(),
	    warmup_cfg.step_gradient_decay(),
	    warmup_cfg.step_sq_gradient_decay(),
	    warmup_cfg.step_stabilization(),
	    warmup_cfg.step_learn_rate_decay()) {
  }

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
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] theta The initial position.
   * @param[in] grad The gradient of the target log density at the initial
   * position.
   * @throw std::invalid_argument If the position and gradient are not the same
   * size.
   */
  MassEstimator(const walnuts::WarmupConfig& warmup_cfg,
		const Vec<S>& theta,
                const Vec<S>& grad)
    : warmup_cfg_(warmup_cfg) {
    validate_same_size(theta, grad, "theta", "grad");

    S smoothing = warmup_cfg.mass_additive_smoothing();
    Vec<S> zero = Vec<S>::Zero(theta.size());
    Vec<S> smooth_vec = Vec<S>::Constant(theta.size(), smoothing);
    Vec<S> sqrt_abs_grad_init = grad.array().abs().sqrt();
    Vec<S> init_prec = (1 - smoothing) * sqrt_abs_grad_init + smooth_vec;
    Vec<S> init_var = init_prec.array().inverse().matrix();
    S dummy_discount = 0.98;  // gets reset before being used
    inv_var_estimator_ = OnlineMoments<S>(dummy_discount, warmup_cfg.mass_init_count(),
                                          zero, init_prec);
    var_estimator_ = OnlineMoments<S>(dummy_discount, warmup_cfg.mass_init_count(), zero, init_var);
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
    double discount_factor = 1.0 - 1.0 / (warmup_cfg_.mass_init_count() + iteration);
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
  /** The warmup configuration for adaptive Walnuts. */
  walnuts::WarmupConfig warmup_cfg_;
  
  /** The online variance estimator for draws. */
  OnlineMoments<S> var_estimator_;

  /** The online inverse variance estimator for scores. */
  OnlineMoments<S> inv_var_estimator_;
};

/**
 * @brief The adaptation handler for the minimum number of micro steps per macro
 * step.
 *
 * After being constructed with a target number of macro steps, this
 * class is given observeations of the number of micro steps taken and
 * adjusts the minimum number of micro steps per macro step in order
 * to achieve the target expected number of macro steps historically.
 * There is slight regularization of a single observation at depth 2,
 * but otherwise it just uses the floor of an average and thus rounds
 * down.
 */
class MinMicroStepsAdaptHandler {
 public:
  /**
   * Construct a minimum number of micro steps per macro step handler.
   *
   * @param[in] expected_macro_steps Expected number of macro steps.
   */
  MinMicroStepsAdaptHandler(double expected_macro_steps)
      : expected_macro_steps_(expected_macro_steps),
        total_macro_steps_(2.0),
        count_(1.0) {}

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
    double min_micro_per_macro =
        std::fmax(1.0, std::floor(mean_micro / expected_macro_steps_));
    std::size_t steps =
        static_cast<std::size_t>(std::round(min_micro_per_macro));
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
 * @tparam Handler Type of adaptation and sampling event handler.
 */
template <class F, typename S, class RNG, class Handler>
class AdaptiveWalnuts {
 public:
  /**
   * @brief Construct an adaptive WALNUTS sampler.
   *
   * The target log density and gradient function must implement the signature
   *
   * ```cpp
   * void normal_logp_grad(const Eigen::Matrix<S, -1, 1>& x,
   *                       S& logp,
   *                       Eigen::Matrix<S, -1, 1>& grad);
   * ```
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
   * @param[in,out] handler Event handler for adaptation and sampling.
   * @param[in] logp_grad The target log density and gradient function.
   * @param[in] theta_init The initial state.
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] sampling_cfg The sampling configuration.
   * @param[in] target_depth The target expected NUTS tree depth.
   */
  AdaptiveWalnuts(RNG& rng,
		  Handler& handler,
		  const F& logp_grad,
		  const Vec<S>& theta_init,
		  const walnuts::InitChainConfig& init_chain_cfg,
		  const walnuts::WarmupConfig& warmup_cfg,
                  const walnuts::SamplingConfig& sampling_cfg,
                  double target_depth = 4.0)
      : init_chain_cfg_(init_chain_cfg),
	warmup_cfg_(warmup_cfg),
	sampling_cfg_(sampling_cfg),
        rand_(rng),
	handler_(handler),
        logp_grad_(logp_grad),
        theta_(theta_init),
        iteration_(0),
        step_adapt_handler_(init_chain_cfg, warmup_cfg),
        mass_estimator_(warmup_cfg, theta_, grad(logp_grad, theta_)),
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
    S logp_select;
    std::size_t depth = 0;
    theta_ = transition_w(
        rand_, logp_grad_, inv_mass, chol_mass, step_adapt_handler_.step_size(),
        sampling_cfg_.max_trajectory_doublings(), sampling_cfg_.max_step_halvings(), 
        min_micro_estimator_.min_micro_steps(), sampling_cfg_.max_hamiltonian_error(),
        std::move(theta_), depth, grad_select, logp_select, step_adapt_handler_);
    mass_estimator_.observe(theta_, grad_select, iteration_);
    min_micro_estimator_.observe(depth);
    handler_.on_warmup(theta_, logp_select, step_size(), inv_mass);  // inv_mass pre, inv_mass() post transition
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
  WalnutsSampler<F, S, RNG, Handler> sampler() {
    handler_.on_warmup_complete(step_size(), inv_mass());
    return WalnutsSampler<F, S, RNG, Handler>(rand_,
					      handler_,
					      logp_grad_.logp_grad_,
					      theta_,
					      mass_estimator_.inv_mass_estimate(),
					      step_adapt_handler_.step_size(),
					      sampling_cfg_.max_trajectory_doublings(), 
					      sampling_cfg_.max_step_halvings(),
					      min_micro_estimator_.min_micro_steps(),
					      sampling_cfg_.max_hamiltonian_error());
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
    return min_micro_estimator_.min_micro_steps();
  }

  std::size_t dim() const noexcept {
    return theta_.size();
  }

  double log_step_size() const noexcept {
    return std::log(step_size());
  }

  Eigen::VectorXd log_mass() const noexcept {
    return inv_mass().array().inverse().log().matrix();
  }

  std::size_t iter() const noexcept {
    return iteration_;
  }
  
 private:
  /** The configuration of initialization for this chain. */
  const walnuts::InitChainConfig init_chain_cfg_;

  /** The warmup configuration. */
  const walnuts::WarmupConfig warmup_cfg_;

  /** The WALNUTS sampler configuration. */
  const walnuts::SamplingConfig sampling_cfg_;

  /** The random number generator required for NUTS. */
  Random<S, RNG> rand_;

  /** The adaptation and sampling event handler. */
  Handler& handler_;

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
