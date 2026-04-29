#pragma once

#include <cmath>
#include <concepts>

#include "util.hpp"

namespace nuts {

/**
 * The Adam stochastic gradient optimizer specialized for step-size
 * adaptation with a decreasing learning rate schedule.

 * The specialization for step size builds in quadratic error,
 * following Nuts. That is, for observed accept rate `acc_obs` and
 * target accept rate `accept_target`, the error is `-0.5 *
 * (accept_target - accept_observed)^2` so that the gradient is
 * `accept_target - accept_observed`.

 * The non-standard effective learning rate schedule divides the
 * learning rate by `pow(t, learn_rate_decay)` in iteration `t`
 * (indexed from 1).  The Robbins-Monro theory around SGD allows
 * values as high as `learn_rate_decay = 1`, and that is a common
 * default, but it can decay too quickly. Nuts used `learn_rate_decay
 * = 0.75` for dual averaging and we have found `learn_rate_decay=0.5`
 * to work well for Adam.
 *
 * @tparam S Type of floating point values.
 */
template <std::floating_point S>
class Adam {
 public:
  /**
   * Construct an Adam optimizer from tuning parameters and initialization.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] accept_rate_target The target acceptance rate.
   * @param[in] learning_rate The learning rate.
   * @param[in] gradient_decay The gradient decay rate.
   * @param[in] sq_gradient_decay The squared gradient decay rate.
   * @param[in] stabilization The estimation stabilization parameter.
   * @param[in] learn_rate_decay The learning rate exponent on iteration.
   */
  Adam(S step_size_init, S accept_rate_target, S learning_rate,
       S gradient_decay, S sq_gradient_decay, S stabilization,
       S learn_rate_decay)
      : theta_(std::log(step_size_init)),
        m_(0),
        v_(0),
        t_(0),
        beta1_pow_(1),
        beta2_pow_(1),
        target_accept_rate_(accept_rate_target),
        learn_rate_(learning_rate),
        beta1_(gradient_decay),
        beta2_(sq_gradient_decay),
        eps_(stabilization),
        learn_rate_decay_(learn_rate_decay) {}

  /**
   * Observe an acceptance probability in (0, 1).
   *
   * @param[in] alpha The acceptance probability.
   */
  void observe(S alpha) noexcept {
    ++t_;
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;

    S grad = target_accept_rate_ - alpha;

    m_ = beta1_ * m_ + (1 - beta1_) * grad;
    v_ = beta2_ * v_ + (1 - beta2_) * grad * grad;

    S m_hat = m_ / (1 - beta1_pow_);
    S v_hat = v_ / (1 - beta2_pow_);

    // dividing by sqrt(t_) non-standard; similar to dual average to make
    S effective_lr = learn_rate_ / std::pow(t_, learn_rate_decay_);
    S denom = std::sqrt(v_hat) + eps_;
    theta_ -= effective_lr * m_hat / denom;
  }

  /**
   * Return the step size estimate.
   *
   * @return The step size.
   */
  S step_size() const noexcept { return std::exp(theta_); }

 private:
  S theta_;
  S m_;
  S v_;
  S t_;
  S beta1_pow_;
  S beta2_pow_;

  const S target_accept_rate_;
  const S learn_rate_;
  const S beta1_;
  const S beta2_;
  const S eps_;
  const S learn_rate_decay_;
};

}  // namespace nuts
