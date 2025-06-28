#pragma once

#include <utility>

#include "walnuts.hpp"
#include "dual_average.hpp"
#include "util.hpp"
#include "online_moments.hpp"

namespace nuts {

/**
 * @brief Return the gradient of the log density at the specified position.
 * 
 * @tparam S Type of scalars.
 * @tparam F Type of the target log density/gradient function.
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
   * tuning paramters.
   *
   * @param[in] mass_init The diagonal of the diagonal initial mass matrix
   * (finite positive scalar components).
   * @param[in] init_count The pseudocount of observations for the
   * initialization (finite positive scalar).
   * @param[in] iter_offset The offset from 1 of the first observation (finite
   * non-negative scalar).
   * @param[in] additive_smoothing The additive smoothing of inverse mass
   * estimates.
   */
  MassAdaptConfig(const Vec<S>& mass_init, S init_count, S iter_offset,
		  S additive_smoothing):
    mass_init_(mass_init), init_count_(init_count), iter_offset_(iter_offset),
    additive_smoothing_(additive_smoothing)
  {}

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
 * @brief The deduction guide for the mass adaptation configuration.
 *
 * The returned type will be `MassAdaptConfig<S>`.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
MassAdaptConfig(const Vec<S>& mass_init, S init_count,
                  S iter_offset)
  -> MassAdaptConfig<S>;

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
  * Construct a step-size adaptation configuration.
  * 
  * @param[in] step_size_init Initial step size (finite positive scalar).
  * @param[in] accept_rate_target Target bidirectional accept rate (scalar 
  * between 0 and 1 exclusive) 
  * @param[in] iter_offset Relative postion of first observation
  * (finite positive scalar).
  * @param[in] learning_rate The learning rate for dual averaging (finite
  * positive scalar).
  * @param[in] decay_rate The decay rate of older observations (finite
  * scalar between 0 and 1 exclusive)
  */
  StepAdaptConfig(S step_size_init, S accept_rate_target, S iter_offset,
		  S learning_rate,  S decay_rate):
      step_size_init_(step_size_init), accept_rate_target_(accept_rate_target),
      iter_offset_(iter_offset), learning_rate_(learning_rate),
      decay_rate_(decay_rate)
  {}

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
 * @brief The deduction guide for step adaptation configuration.
 * 
 * The deduced type is `StepAdaptConfig<S>`.
 * 
 * @tparam S Type of scalars.
 */
template <typename S>
StepAdaptConfig(S step_size_init, S accept_rate_target, S iter_offset,
		S learning_rate, S decay_rate)
  -> StepAdaptConfig<S>;
  

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
   * macro leapfrog step (finite positive scalar).
   * @param[in] max_nuts_depth The maximum number of trajectory
   * doublings for NUTS (finite positive integer).
   * @param[in] max_step_depth The maximum number of step doublings
   * per macro step (finite non-negative integer).
   */
  WalnutsConfig(S log_max_error,
                Integer max_nuts_depth,
                Integer max_step_depth):
      log_max_error_(log_max_error), max_nuts_depth_(max_nuts_depth),
      max_step_depth_(max_step_depth)
  {}

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
 * WALNUTS works through callbacks to an adaptation handler, impelemented
 * as a functor through the method `operator()(S)`.  This handler maintains
 * the dual averaging adaptation and also returns the current step size
 * estimate through the method `step_size()`.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class StepAdaptHandler {
 public:
  StepAdaptHandler(S step_size_init, S target_accept_rate, S iter_offset,
                   S learning_rate, S decay_rate):
      dual_average_(step_size_init, target_accept_rate, iter_offset,
                    learning_rate, decay_rate)
  {}

  /**
   * @brief Update with the estimate of step size given the specified
   * acceptance probability.
   *
   * @param accept_prob The observed acceptance probability.
   */
  void operator()(S accept_prob) {
    dual_average_.observe(accept_prob);
  }

  /**
   * @brief Return the estimated step size.
   *
   * @return The estimated step size.
   */
  S step_size() {
    return dual_average_.epsilon();
  }

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
   * with the function `observe()`.  At each step, the discount factor for discounting
   * past draws the online moment estimators is set to
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
   * the additive smoothign and adding the additive smoothing.
   *
   * @param mass_cfg The mass matrix adaptation configuration.
   * @param theta The initial position.
   * @param grad The gradient of the target log density at the initial position.
   */
  MassEstimator(const MassAdaptConfig<S>& mass_cfg, const Vec<S>& theta,
		const Vec<S>& grad):
    mass_cfg_(mass_cfg),
    var_estimator_(0, theta.size()),
    inv_var_estimator_(0, theta.size())
  {
    S smoothing = mass_cfg_.additive_smoothing_;
    Vec<S> zero = Vec<S>::Zero(theta.size());
    Vec<S> smooth_vec = Vec<S>::Constant(theta.size(), smoothing);
    Vec<S> sqrt_abs_grad_init = grad.array().abs().sqrt();
    Vec<S> init_prec = (1 - smoothing) * sqrt_abs_grad_init + smooth_vec;
    Vec<S> init_var = init_prec.array().inverse().matrix();
    S dummy_discount = 0.98;  // gets reset before being used
    inv_var_estimator_ = OnlineMoments<S>(dummy_discount, mass_cfg.iter_offset_,
					  zero, init_prec);
    var_estimator_ = OnlineMoments<S>(dummy_discount, mass_cfg.iter_offset_,
				      zero, init_var);
  }

  /**
   * @brief Update the estimate for the specified iteration with the
   * observation and gradient.
   *
   * @param theta The position observed.
   * @param grad The gradient of the log density at the position.
   * @param iteration The iteration number (non-negative integer).
   */
  void observe(const Vec<S>& theta, const Vec<S>& grad, Integer iteration) {
    double discount_factor = 1.0 - 1.0 / (mass_cfg_.iter_offset_ + iteration);
    var_estimator_.set_discount_factor(discount_factor);  // TODO: one encapsulated function
    var_estimator_.observe(theta);
    inv_var_estimator_.set_discount_factor(discount_factor);
    inv_var_estimator_.observe(grad); 
  }

  /**
   * @brief Return an estimate of the inverse mass matrix.
   *
   * @return The inverse mass matrix estimate.
   */
  Vec<S> inv_mass_estimate() {
    Vec<S> inv_mass_est_var = var_estimator_.variance().array();
    Vec<S> inv_mass_est_inv_var = inv_var_estimator_.variance().array().inverse()
      .matrix();
    Vec<S> inv_mass_est = (inv_mass_est_var.array() * inv_mass_est_inv_var.array())
      .sqrt().matrix();
    return inv_mass_est;
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
  AdaptiveWalnuts(RNG& rng,
                  F& logp_grad,
                  const Vec<S>& theta_init,
                  const MassAdaptConfig<S>& mass_cfg,
                  const StepAdaptConfig<S>& step_cfg,
                  const WalnutsConfig<S>& walnuts_cfg):
    mass_cfg_(mass_cfg), 
    step_cfg_(step_cfg),
    walnuts_cfg_(walnuts_cfg),
    rand_(rng),
    logp_grad_(logp_grad),
    theta_(theta_init),
    iteration_(0),
    step_adapt_handler_(step_cfg.step_size_init_, step_cfg.accept_rate_target_,
			step_cfg.iter_offset_, step_cfg.learning_rate_,
			step_cfg.decay_rate_),
    mass_estimator_(mass_cfg_, theta_, grad(logp_grad, theta_))
  {}



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
    theta_ = transition_w(rand_, logp_grad_, inv_mass, chol_mass,
        		  step_adapt_handler_.step_size(),
                          walnuts_cfg_.max_nuts_depth_,
                          std::move(theta_), grad_select, walnuts_cfg_.log_max_error_,
                          step_adapt_handler_);
    mass_estimator_.observe(theta_, grad_select, iteration_);
    ++iteration_;
    return theta_;
  }

  /**
   * @brief Return a WALNUTS sampler with the current tuning parameter
   * estimates. 
   *
   * The returned sampler forms a proper Markov chain.
   *
   * @return The WALNUTS sampler with current tuning parameter estimates. 
   */
  WalnutsSampler<F, S, RNG> sampler() {
    return WalnutsSampler<F, S, RNG>(
        rand_,
        logp_grad_,
        theta_,
        mass_estimator_.inv_mass_estimate(),
        step_adapt_handler_.step_size(),
        walnuts_cfg_.max_nuts_depth_,
        walnuts_cfg_.log_max_error_);
  }

 private:
  /** The mass adaptaiton configuration. */
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

} // namespace nuts
