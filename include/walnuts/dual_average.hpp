#pragma once

#include <cmath>
#include <stdexcept>

namespace nuts {

/**
 * Class holding configuration and state for a dual averaging
 * estimator.
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
   * The algorithm converges at a logarithmic rate.  It should be
   * reasonably accurate at achieving a target acceptance rate, but
   * the step size estimates themselves tend to be higher variance.
   *
   * @param epsilon_init Initial value (> 0).
   * @param target_accept_rate Target acceptance rate (> 0).
   * @param obs_count_offset Iteration offset (> 0, default 10).
   * @param learn_rate Learning rate (> 0, default 0.05).
   * @param decay_rate Decay for averaging (> 0, default 0.75).
   * @throw std::except If initial epsilon is not finite and positive.
   * @throw std::except If target acceptance rate is not finite and positive.
   * @throw std::except If iteration offset is not finite and positive.
   * @throw std::except If learning rate is not finite and positive.
   */
  DualAverage(S epsilon_init, S target_accept_rate, S obs_count_offset = 10, S learn_rate = 0.05,
              S decay_rate = 0.75):
      log_est_(std::log(epsilon_init)),
      log_est_avg_(0),
      grad_avg_(0),
      obs_count_(1),
      log_step_offset_(std::log(10) + std::log(epsilon_init)),
      target_accept_rate_(target_accept_rate),
      obs_count_offset_(obs_count_offset),
      learn_rate_(learn_rate),
      decay_rate_(decay_rate) {
    if (!(epsilon_init > 0 && std::isfinite(epsilon_init))) {
      throw std::invalid_argument("Initial epsilon must be positive and finite.");
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
   * @param alpha Observed value (> 0).
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
   * @brief Return the current estimate.
   *
   * @return The current estimate.
   */
  inline S epsilon() const noexcept {
    return std::exp(log_est_avg_);
  }

private:
  /** The local estimate. */
  S log_est_;

  /** The log estimate, a decayed running average of `log_est_`. */
  S log_est_avg_;

  /** Average gradient (hbar in paper). */
  S grad_avg_;

  /** The observation count. */
  S obs_count_;

  /** The target log-step offset. */
  const S log_step_offset_;

  /** The target acceptance rate. */
  const S target_accept_rate_;

  /** The observation count offset. */
  const S obs_count_offset_;

  /** Learning rate. */
  const S learn_rate_;

  /** The decay rate. */
  const S decay_rate_;
};

}  // namespace nuts
