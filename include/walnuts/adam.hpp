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
 * @brief The immutable configuration for Adam-based step-size adaptation.
 *
 * @tparam S The type of scalars.
 */
template <typename S>
struct AdamConfig {
  /**
   * @brief Construct a step-size adaptation configuration given the
   * tuning parameters.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] target_accept_rate The target acceptance rate.
   * @param[in] learn_rate The learning rate.
   * @param[in] beta1 The decay rate of the moving average for gradients.
   * @param[in] beta2 The decay rate of the moving average for squared
   * gradients.
   * @param[in] epsilon The stabilization constant added to the denominator of
   * updates.
   * @throw std::invalid_argument If the initial step size is not finite and
   * positive.
   * @throw std::invalid_argument If the learning rate is not positive and
   * finite.
   * @throw std::invalid_argument If `beta1` is not in (0, 1).
   * @throw std::invalid_argument If `beta2` is not in (0, 1).
   * @throw std::invalid_argument If `epsilon" is not positive and finite.
   */
  AdamConfig(S step_size_init, S target_accept_rate, S learn_rate = 0.2,
             S beta1 = 0.3, S beta2 = 0.99, S epsilon = 1e-4)
      : step_size_init_(step_size_init),
        target_accept_rate_(target_accept_rate),
        learn_rate_(learn_rate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {
    validate_positive(step_size_init, "step_size_init");
    validate_probability(target_accept_rate, "target_accept_rate");
    validate_positive(learn_rate, "learn_rate");
    validate_probability(beta1, "beta1");
    validate_probability(beta2, "beta2");
    validate_positive(epsilon, "epsilon");
  }

  /** The initial macro step size. */
  const S step_size_init_;

  /** The target expected Metropolis acceptance rate of macro steps. */
  const S target_accept_rate_;

  /** The learning rate for Adam. */
  const S learn_rate_;

  /** The decay rate of the moving average for gradients */
  const S beta1_;

  /** The decay rate of the moving average for squared gradients. */
  const S beta2_;

  /** The additive stabiliziation constant for the denominator of updates. */
  const S epsilon_;
};

/**
 * The Adam stochastic gradient optimizer specialized for step-size adaptation.
 */
template <typename S>
class Adam {
 public:
  /**
   * Construct an Adam optimizer from a configuration.
   *
   * @param[in] cfg The configuration for the optimizer.
   */
  Adam(const AdamConfig<S>& cfg)
      : theta_(std::log(cfg.step_size_init_)),
        m_(0),
        v_(0),
        t_(0),
        beta1_pow_(1),
        beta2_pow_(1),
        target_accept_rate_(cfg.target_accept_rate_),
        learn_rate_(cfg.learn_rate_),
        beta1_(cfg.beta1_),
        beta2_(cfg.beta2_),
        eps_(cfg.epsilon_) {}

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
