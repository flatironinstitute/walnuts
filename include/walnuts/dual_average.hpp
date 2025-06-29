#pragma once

#include <cmath>
#include <stdexcept>

namespace nuts {

/**
 * @brief The configuration and state for a dual averaging estimator
 * of step size based on a target acceptance rate.
 *
 * The estimator works by observing empirical acceptance rates.  If
 * they are above the target acceptance rate, the step size is
 * increased; if they are below, the step size is decreased.  This
 * implies a standard normal error model as the basis for the
 * stochastic gradients. Estimates are averaged as new observations
 * come in for stability.
 *
 * The algorithm is defined in the following reference.
 *
 * \see Hoffman, M.D. and Gelman, A. 2014. The No-U-Turn sampler:
 * Adaptively setting path lengths in Hamiltonian Monte
 * Carlo. *Journal of Machine Learning Research*, 15(1), pp.1593-1623.
 *
 * @tparam S Type of scalars.
 */
template <typename S>
class DualAverage {
public:
  /**
   * @brief Construct a dual averaging estimator with the specified initial
   * value, target value, and tuning parameters.
   *
   * Larger values of `obs_count_offset` delay fast adaptation and damp early
   * instability.  Smaller values of `learn_rate` lead to slower, but more
   * stable adaptation.  Higher values of `decay_rate` slow the freezing of
   * updates and provide more smoothing.
   *
   * The dual averaging algorithm converges at a logarithmic rate.  It
   * should be reasonably accurate at achieving a target acceptance
   * rate, but the step size estimates themselves tend to be higher
   * variance.
   *
   * @param[in] step_size_init Initial step size value.
   * @param[in] target_accept_rate Target acceptance rate.
   * @param[in] obs_count_offset Iteration offset.
   * @param[in] learn_rate Learning rate.
   * @param[in] decay_rate Decay for averaging.
   * @throw std::except If initial step_size is not finite and positive.
   * @throw std::except If target acceptance rate is not finite and positive.
   * @throw std::except If iteration offset is not finite and positive.
   * @throw std::except If learning rate is not finite and positive.
   * @pre step_size_init > 0
   * @pre target_accept_rate > 0
   * @pre obs_count_offset > 0
   * @pre learn_rate > 0
   * @pre decay_rate > 0
   */
  DualAverage(S step_size_init, S target_accept_rate, S obs_count_offset = 10,
	      S learn_rate = 0.05, S decay_rate = 0.75):
      log_est_(std::log(step_size_init)),
      log_est_avg_(0),
      grad_avg_(0),
      obs_count_(1),
      log_step_offset_(std::log(10) + std::log(step_size_init)),
      target_accept_rate_(target_accept_rate),
      obs_count_offset_(obs_count_offset),
      learn_rate_(learn_rate),
      decay_rate_(decay_rate) {
    if (!(step_size_init > 0 && std::isfinite(step_size_init))) {
      throw std::invalid_argument("Initial step_size must be positive and finite.");
    }
    if (!(target_accept_rate > 0 && std::isfinite(target_accept_rate))) {
      throw std::invalid_argument("Target acceptance rate must be positive and finite.");
    }
    if (!(obs_count_offset > 0 && std::isfinite(obs_count_offset))) {
      throw std::invalid_argument("Iteration offset must be positive and finite.");
    }
    if (!(learn_rate > 0 && std::isfinite(learn_rate))) {
      throw std::invalid_argument("Learning rate must be positive and finite.");
    }
    if (!(decay_rate > 0 && std::isfinite(decay_rate))) {
      throw std::invalid_argument("Decay rate must be positive and finite.");
    }
  }

  /**
   * @brief Update the state for the observed value.
   *
   * @param[in] alpha The observed value.
   * @pre alpha > 0
   */
  inline void observe(S alpha) noexcept {
    S prop = 1 / (obs_count_ + obs_count_offset_);
    grad_avg_ = (1 - prop) * grad_avg_ + prop * (target_accept_rate_ - alpha);
    S last_log_est = log_est_;
    log_est_ = log_step_offset_ - std::sqrt(obs_count_) / learn_rate_ * grad_avg_;
    S prop2 = std::pow(obs_count_, -decay_rate_); 
    log_est_avg_ = prop2 * log_est_
        + (1 - prop2) * last_log_est;
    ++obs_count_;
  }

  /**
   * @brief Return the current step-size estimate.
   *
   * @return The current estimate of step size.
   */
  inline S step_size() const noexcept {
    return std::exp(log_est_avg_);
  }

private:
  /** The local estimate of step size last iteration. */
  S log_est_;

  /** The log estimate of step size, a decayed running average of `log_est_`. */
  S log_est_avg_;

  /** The average gradient. */
  S grad_avg_;

  /** The observation count. */
  S obs_count_;

  /** The target log-step offset. */
  const S log_step_offset_;

  /** The target acceptance rate. */
  const S target_accept_rate_;

  /** The observation count offset. */
  const S obs_count_offset_;

  /** The learning rate. */
  const S learn_rate_;

  /** The decay rate. */
  const S decay_rate_;
};

}  // namespace nuts
