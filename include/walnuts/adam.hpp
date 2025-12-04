#pragma once

#include <cmath>
#include <stdexcept>

namespace nuts {


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
struct AdamConfig {
  /**
   * @brief Construct a step-size adaptation configuration given the
   * tuning parameters.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] target_accept_rate The target acceptance rate.
   * @param[in] learn_rate The learning rate for Adam.
   * @param[in] beta1 The beta1 tuning parameter for Adam.
   * @param[in] beta2 The beta2 tuning parameter for Adam.
   * @param[in] epsilon The epsilon tuning parameter for Adam.
   * @throw std::invalid_argument If the initial step size is not finite and
   * positive.
   * @throw std::invalid_argument If the learning rate is not positive and
   * finite. 
   * @throw std::invalid_argument If `beta1` is not in (0, 1).
   * @throw std::invalid_argument If `beta2` is not in (0, 1).
   * @throw std::invalid_argument If `epsilon" is not positive and finite.
   */
  AdamConfig(S step_size_init, S target_accept_rate,
	     S learn_rate = 0.2, S beta1 = 0.3, S beta2 = 0.99, S epsilon = 1e-4)
      : step_size_init_(step_size_init),
        target_accept_rate_(target_accept_rate),
	learn_rate_(learn_rate),
	beta1_(beta1),
	beta2_(beta2),
	epsilon_(epsilon) {
    if (!(step_size_init > 0) || std::isinf(step_size_init)) {
      throw std::invalid_argument("Initial count must be positive and finite.");
    }
    if (!(target_accept_rate > 0) || !(target_accept_rate < 1)) {
      throw std::invalid_argument("Acceptance rate target must be in (0, 1)");
    }
    if (!(learn_rate > 0) || std::isinf(learn_rate)) {
      throw std::invalid_argument("Learning rate must be positive and finite.");
    }
    if (!(beta1 > 0) || !(beta1 < 1)) {
      throw std::invalid_argument("beta1 must be in (0, 1)");
    }
    if (!(beta2 > 0) || !(beta2 < 1)) {
      throw std::invalid_argument("beta2 must be in (0, 1)");
    }
    if (!(epsilon > 0) || std::isinf(epsilon)) {
      throw std::invalid_argument("epsilon must be positive and finite.");
    }
  }

  /** The initial macro step size. */
  const S step_size_init_;

  /** The target expected Metropolis acceptance rate of macro steps. */
  const S target_accept_rate_;

  /** The learning rate for Adam. */
  const S learn_rate_;

  /** The beta1 parameter for Adam */
  const S beta1_;

  /** The beta2 parameter for Adam. */
  const S beta2_;

  /** The epsilon parameter for Adam. */
  const S epsilon_;
};
  
template <typename S>
class Adam {
 public:
  Adam(const AdamConfig<S>& cfg) :
    theta_(std::log(cfg.step_size_init_)),
    m_(0),
    v_(0),
    t_(0),
    beta1_pow_(1),
    beta2_pow_(1),
    target_accept_rate_(cfg.target_accept_rate_),
    learn_rate_(cfg.learn_rate_),
    beta1_(cfg.beta1_),
    beta2_(cfg.beta2_),
    eps_(cfg.epsilon_) {
  }

  inline void observe(S alpha) noexcept {
    ++t_;
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;

    const S grad = target_accept_rate_ - alpha;

    m_ = beta1_ * m_ + (1 - beta1_) * grad;
    v_ = beta2_ * v_ + (1 - beta2_) * grad * grad;

    const S m_hat = m_ / (1 - beta1_pow_);
    S v_hat = v_ / (1 - beta2_pow_);

    if (v_hat < 0) {
      v_hat = 0;
    }

    const S denom = std::sqrt(v_hat) + eps_;
    const S effective_lr = learn_rate_ / std::sqrt(static_cast<S>(t_));
    theta_ -= effective_lr * m_hat / denom;
  }

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
