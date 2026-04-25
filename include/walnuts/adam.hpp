#pragma once

#include <cmath>

#include "util.hpp"

namespace nuts {

/**
 * Return the rectified linear unit (relu) of the argument.
 *
 * relu(x) = x if x > 0 and relu(x) = 0 if x <= 0`
 *
 * @param[in] x The argument.
 * @return The relu of the argument.
 */
template <typename S>
S relu(S x) {
  return x < 0 ? 0 : x;
}

/**
 * The Adam stochastic gradient optimizer specialized for step-size adaptation.
 */
template <typename S>
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
   */
  Adam(double step_size_init, double accept_rate_target,
       double learning_rate, double gradient_decay, double sq_gradient_decay,
       double stabilization)
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
      eps_(stabilization) {}
  
  /**
   * Observe an acceptance probabilty in (0, 1).
   *
   * @param[in] alpha The acceptance probability.
   */
  inline void observe(S alpha) noexcept {
    ++t_;
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;

    const S grad = target_accept_rate_ - alpha;

    m_ = beta1_ * m_ + (1 - beta1_) * grad;
    v_ = beta2_ * v_ + (1 - beta2_) * grad * grad;

    const S m_hat = m_ / (1 - beta1_pow_);
    S v_hat = relu(v_ / (1 - beta2_pow_));

    const S denom = std::sqrt(v_hat) + eps_;
    const S effective_lr = learn_rate_ / std::sqrt(static_cast<S>(t_));
    theta_ -= effective_lr * m_hat / denom;
  }

  /**
   * Return the step size estimate.
   *
   * @return The step size.
   */
  inline S step_size() const noexcept { return std::exp(theta_); }

 private:
  S theta_;
  S m_;
  S v_;
  std::size_t t_;
  S beta1_pow_;
  S beta2_pow_;

  const S target_accept_rate_;
  const S learn_rate_;
  const S beta1_;
  const S beta2_;
  const S eps_;
};

}  // namespace nuts
